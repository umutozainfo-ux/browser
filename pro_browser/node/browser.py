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
        self.frame_event = asyncio.Event()

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
                
                self.running = True
                await self.start_screencast()
                
                logger.info("Navigating to initial page...")
                try:
                    await self.page.goto('https://www.google.com', wait_until="domcontentloaded", timeout=15000)
                except Exception as nav_e:
                    logger.warning(f"Initial navigation slow/failed: {nav_e}")
                
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
                'quality': 30, # Low quality for speed as requested
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
        
        session_id = data.get('sessionId')
        asyncio.create_task(self._safe_ack(session_id))
        
        raw_data = data.get('data')
        # Duplicate detection (Send only changed)
        if hasattr(self, '_last_jpeg') and self._last_jpeg == raw_data:
            return
            
        self._last_jpeg = raw_data # Store raw B64 for MJPEG mode
        
        try:
            # Notify waiters (both MJPEG and H264 loops wait on this)
            self.frame_event.set()
            self.frame_event.clear()
            
            # Offload heavy decoding for H.264 path only if needed?
            # Actually, let's just trigger the event. The H.264 loop will call get_latest_raw which triggers decoding if null.
            # But wait, we need to decode to get the 'frame' object for H.264. 
            # Let's fire-and-forget the decoder for H.264 readiness, but MJPEG can grab raw bytes instantly.
            loop = asyncio.get_running_loop()
            loop.create_task(self._process_frame_in_thread(raw_data))
        except Exception as e:
             logger.error(f"Frame dispatch error: {e}")

    async def _process_frame_in_thread(self, b64_data):
        try:
            # Run CPU-bound work in executor
            loop = asyncio.get_running_loop()
            frame = await loop.run_in_executor(None, self._decode_frame, b64_data)
            if frame: self.last_frame = frame
        except Exception as e:
            logger.error(f"Async frame processing error: {e}")

    def _decode_frame(self, b64_data):
        # This runs in a separate thread
        try:
            image_data = base64.b64decode(b64_data)
            import cv2
            nparr = np.frombuffer(image_data, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_np is None: return None
            
            # Optimization: Keep in BGR
            frame = VideoFrame.from_ndarray(img_np, format='bgr24')
            from fractions import Fraction
            frame.pts = int(time.time() * 90000)
            frame.time_base = Fraction(1, 90000)
            return frame
        except Exception as e:
            return None

    async def _safe_ack(self, session_id):
        if not self.running or not self.cdp or not session_id: return
        try:
            await self.cdp.send('Page.screencastFrameAck', {'sessionId': session_id})
        except:
            pass

    async def get_latest_frame(self) -> Optional[VideoFrame]:
        """Wait for the next frame efficiently"""
        await self.frame_event.wait()
        return self.last_frame

    def force_refresh(self):
        """Force next frame to be processed even if duplicate"""
        if hasattr(self, '_last_jpeg'):
             delattr(self, '_last_jpeg')
        if hasattr(self, '_last_raw_data'): # Cleanup
             delattr(self, '_last_raw_data')

    def get_latest_jpeg(self):
        """Get raw JPEG bytes (base64 or binary) for MJPEG stream"""
        if hasattr(self, '_last_jpeg') and self._last_jpeg:
            # Return raw base64 string directly - simplest for Text websocket, 
            # but binary is better. Let's return bytes.
            return base64.b64decode(self._last_jpeg)
        return None

    def get_latest_raw(self):
        """Return the latest frame as a numpy array with timestamp, avoiding async overhead"""
        if self.last_frame:
            # It's already BGR24 in our optimization
            return self.last_frame.to_ndarray(format='bgr24'), self.last_frame.pts
        return None, None

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
                elif type == 'wheel':
                    await self.page.mouse.wheel(delta_x=event.get('deltaX', 0), delta_y=event.get('deltaY', 0))
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
