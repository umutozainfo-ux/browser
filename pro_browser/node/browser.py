import asyncio
import base64
import logging
import time
from typing import Optional

from playwright.async_api import async_playwright
import numpy as np
from av import VideoFrame

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Browser")

class BrowserManager:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.cdp = None
        self.last_frame: Optional[VideoFrame] = None
        self.running = False
        self.width = 1280
        self.height = 720
        self.lock = asyncio.Lock()
        self.input_lock = asyncio.Lock()

    async def start(self, proxy: Optional[dict] = None):
        logger.info("Wait for lock to START browser...")
        async with self.lock:
            if self.running: 
                logger.info("Browser already running.")
                return
            logger.info(f"Acquired lock. Starting Browser (Proxy: {proxy['server'] if proxy else 'None'})...")
            try:
                logger.info("Initializing Playwright...")
                self.playwright = await async_playwright().start()
                
                logger.info("Launching Chromium...")
                launch_args = [
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-infobars',
                    '--window-size=1280,720',
                    '--disable-blink-features=AutomationControlled'
                ]
                
                self.browser = await self.playwright.chromium.launch(
                    headless=True,
                    args=launch_args,
                    proxy=proxy
                )
                
                logger.info("Creating Context & Stealthing...")
                self.context = await self.browser.new_context(
                    viewport={'width': self.width, 'height': self.height},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                from stealth import STEALTH_SCRIPTS
                for script in STEALTH_SCRIPTS:
                    await self.context.add_init_script(script)

                logger.info("Opening Page...")
                self.page = await self.context.new_page()
                self.cdp = await self.context.new_cdp_session(self.page)
                
                logger.info("Navigating to initial page...")
                try:
                    await self.page.goto('https://www.google.com', wait_until="domcontentloaded", timeout=15000)
                except Exception as nav_e:
                    logger.error(f"Initial navigation failed: {nav_e}")
                    if "ERR_TUNNEL_CONNECTION_FAILED" in str(nav_e) or "ERR_PROXY_CONNECTION_FAILED" in str(nav_e):
                        raise Exception(f"PROXY_FAILURE: {nav_e}")
                    raise nav_e
                
                await self.start_screencast()
                
                self.running = True
                logger.info("Browser started successfully.")
            except Exception as e:
                logger.error(f"Failed to start browser: {e}")
                await self._cleanup()
                raise e # Re-raise to let the caller handle it
        logger.info("Released lock after START attempt.")

    async def start_screencast(self):
        """Start CDP Screencast for high FPS"""
        if not self.cdp: return
        try:
            logger.info("Enabling CDP Page domain...")
            await self.cdp.send('Page.enable')
            
            logger.info("Starting CDP Screencast...")
            await self.cdp.send('Page.startScreencast', {
                'format': 'jpeg',
                'quality': 60,
                'maxWidth': self.width,
                'maxHeight': self.height,
                'everyNthFrame': 1
            })
            self.cdp.on('Page.screencastFrame', self._on_screencast_frame)
            logger.info("CDP Screencast started successfully.")
        except Exception as e:
            logger.error(f"Error starting screencast: {e}")

    def is_healthy(self) -> bool:
        if not self.running: return False
        try:
            return self.browser and self.browser.is_connected()
        except:
            return False

    def _on_screencast_frame(self, data):
        """Handle incoming CDP frame with EPIPE protection"""
        if not self.running or not self.cdp: return
        try:
            asyncio.create_task(self._safe_ack(data.get('sessionId')))
            if not data.get('data'): return
            
            # Log every 100th frame instead of every frame
            if not hasattr(self, '_frame_count'): self._frame_count = 0
            self._frame_count += 1
            if self._frame_count % 100 == 1:
                logger.info(f"Screencast frame received ({self._frame_count})")

            image_data = base64.b64decode(data['data'])
            import cv2
            nparr = np.frombuffer(image_data, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_np is None: return
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            frame = VideoFrame.from_ndarray(img_rgb, format='rgb24')
            from fractions import Fraction
            frame.pts = int(time.time() * 90000)
            frame.time_base = Fraction(1, 90000)
            self.last_frame = frame
        except Exception as e:
            if "EPIPE" not in str(e):
                logger.error(f"Frame handling error: {e}", exc_info=True)

    async def _safe_ack(self, session_id):
        if not self.running or not self.cdp or not session_id: return
        try:
            await self.cdp.send('Page.screencastFrameAck', {'sessionId': session_id})
        except:
            pass

    async def get_latest_frame(self) -> Optional[VideoFrame]:
        return self.last_frame

    async def handle_input(self, event):
        if not self.running or not self.page: return
        async with self.input_lock:
            try:
                type = event.get('type')
                if type == 'mousemove':
                    await self.page.mouse.move(event['x'], event['y'])
                elif type == 'mousedown':
                    await self.page.mouse.move(event['x'], event['y'])
                    await self.page.mouse.down()
                elif type == 'mouseup':
                    await self.page.mouse.move(event['x'], event['y'])
                    await self.page.mouse.up()
                elif type == 'keydown':
                    await self.page.keyboard.press(event['key'])
                elif type == 'navigate':
                    await self.navigate_to(event.get('url'))
                elif type == 'back':
                    await self.page.go_back()
                elif type == 'forward':
                    await self.page.go_forward()
                elif type == 'reload':
                    await self.page.reload()
            except Exception as e:
                logger.error(f"Input error: {e}", exc_info=True)

    async def navigate_to(self, url: str):
        if not self.page or not url: return
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            await self.page.goto(url, wait_until="domcontentloaded", timeout=10000)
        except Exception as e:
            logger.error(f"Navigation error: {e}")

    async def _cleanup(self):
        """Internal cleanup without locking to avoid deadlocks"""
        logger.info("Cleaning up browser resources...")
        self.running = False
        try:
            if self.cdp:
                try: await self.cdp.detach()
                except: pass
            if self.page:
                try: await self.page.close()
                except: pass
            if self.context:
                try: await self.context.close()
                except: pass
            if self.browser:
                try: await self.browser.close()
                except: pass
            if self.playwright:
                try: await self.playwright.stop()
                except: pass
        except Exception as e:
            logger.error(f"Error during internal cleanup: {e}")
        finally:
            self.page = None
            self.context = None
            self.browser = None
            self.playwright = None
            self.cdp = None
            self.last_frame = None

    async def close(self):
        logger.info("Wait for lock to CLOSE browser...")
        async with self.lock:
            logger.info("Acquired lock for CLOSE.")
            await self._cleanup()
        logger.info("Released lock after CLOSE.")
