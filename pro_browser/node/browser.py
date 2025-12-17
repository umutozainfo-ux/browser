import asyncio
import base64
import logging
import time
from typing import Optional

from playwright.async_api import async_playwright
import numpy as np
from av import VideoFrame

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

    async def start(self):
        logger.info("Starting Browser...")
        self.playwright = await async_playwright().start()
        # Launch options for performance and stealth
        self.browser = await self.playwright.chromium.launch(
            headless=False, # Headless=False is better for stealth usually, but we can try True for docker
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-infobars',
                '--window-size=1280,720',
                '--disable-blink-features=AutomationControlled'
            ]
        )
        self.context = await self.browser.new_context(
            viewport={'width': self.width, 'height': self.height},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        # Apply Stealth
        from stealth import STEALTH_SCRIPTS
        for script in STEALTH_SCRIPTS:
            await self.context.add_init_script(script)

        self.page = await self.context.new_page()
        self.cdp = await self.context.new_cdp_session(self.page)
        
        await self.start_screencast()
        await self.page.goto('https://www.google.com')
        self.running = True

    async def start_screencast(self):
        """Start CDP Screencast for high FPS"""
        await self.cdp.send('Page.startScreencast', {
            'format': 'jpeg',
            'quality': 60,
            'maxWidth': self.width,
            'maxHeight': self.height,
            'everyNthFrame': 1
        })
        self.cdp.on('Page.screencastFrame', self._on_screencast_frame)

    def _on_screencast_frame(self, data):
        """Handle incoming CDP frame"""
        try:
            # data['data'] is base64 encoded jpeg
            # We need to acknowledge the frame to keep the stream flowing
            asyncio.create_task(self.cdp.send('Page.screencastFrameAck', {'sessionId': data['sessionId']}))
            
            # Decode JPEG to AV VideoFrame
            # This is CPU intensive, might want to optimize
            image_data = base64.b64decode(data['data'])
            
            # Use PyAV or OpenCV to convert to frame
            # For simplicity in aiortc, we construct a VideoFrame from numpy array
            import cv2
            nparr = np.frombuffer(image_data, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB (OpenCV uses BGR)
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            frame = VideoFrame.from_ndarray(img_rgb, format='rgb24')
            # Use standard time base
            from fractions import Fraction
            frame.pts = int(time.time() * 90000)
            frame.time_base = Fraction(1, 90000)
            self.last_frame = frame
            
        except Exception as e:
            logger.error(f"Frame handling error: {e}")

    async def get_latest_frame(self) -> Optional[VideoFrame]:
        return self.last_frame

    # Input Handling
    async def handle_input(self, event):
        if not self.page: return
        
        try:
            type = event.get('type')
            if type == 'mousemove':
                await self.page.mouse.move(event['x'], event['y'])
            elif type == 'mousedown':
                await self.page.mouse.down()
            elif type == 'mouseup':
                await self.page.mouse.up()
            elif type == 'keydown':
                await self.page.keyboard.press(event['key'])
        except Exception as e:
            logger.error(f"Input error: {e}")

    async def close(self):
        self.running = False
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
