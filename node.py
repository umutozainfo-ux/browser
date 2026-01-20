import os
import uuid
import time
import json
import asyncio
import base64
import platform
import httpx
import cv2
import numpy as np
import threading
import logging
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer, RTCDataChannel
from av import VideoFrame
import fractions

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("StealthNode")

class Config:
    NODE_ID = os.getenv("NODE_ID", f"node-{str(uuid.uuid4())[:6]}")
    SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
    NODE_HOST = os.getenv("NODE_HOST", "localhost")
    NODE_PORT = int(os.getenv("NODE_PORT", 8001))
    HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"
    MAX_BROWSERS = int(os.getenv("MAX_BROWSERS", 20))
    HEARTBEAT_INTERVAL = 5
    
    WIDTH = 1280
    HEIGHT = 720
    FPS = 30
    JPEG_QUALITY = 50

img_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

STEALTH_JS = r"""
(() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(p) {
        if (p === 37445) return 'Google Inc. (Intel)';
        if (p === 37446) return 'ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)';
        return getParameter.apply(this, arguments);
    };
    Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
    Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
    const mockPlugins = [
        { name: 'PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
        { name: 'Chrome PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' }
    ];
    Object.defineProperty(navigator, 'plugins', { get: () => mockPlugins });
    
    // Smooth scrolling fix
    window.addEventListener('wheel', (e) => {
        if (e.ctrlKey) e.preventDefault();
    }, { passive: false });
})();
"""

class BrowserVideoTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=1)

    async def recv(self):
        frame = await self._queue.get()
        return frame

    def push_frame(self, frame):
        if self._queue.full():
            try: self._queue.get_nowait()
            except: pass
        try: self._queue.put_nowait(frame)
        except: pass

class BrowserInstance:
    def __init__(self, bid: str):
        self.id = bid
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.is_active = False
        self.websockets: List[WebSocket] = []
        self.pcs: List[RTCPeerConnection] = []
        self.tracks: List[BrowserVideoTrack] = []
        self._last_pts = 0
        self._frames_sent = 0
        self._last_frame_log = time.time()

    async def launch(self):
        logger.info(f"Launching browser {self.id}...")
        self.playwright = await async_playwright().start()
        
        args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--window-position=0,0",
            "--disable-infobars",
            "--mute-audio",
            "--disable-canvas-aa",
            "--disable-2d-canvas-clip-utils",
            "--use-gl=desktop" if platform.system() != "Windows" else "--disable-gpu",
        ]

        self.browser = await self.playwright.chromium.launch(
            headless=Config.HEADLESS,
            args=args,
            channel="chrome"
        )
        
        self.context = await self.browser.new_context(
            viewport={"width": Config.WIDTH, "height": Config.HEIGHT},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        await self.context.add_init_script(STEALTH_JS)
        self.page = await self.context.new_page()
        
        self.cdp = await self.context.new_cdp_session(self.page)
        logger.info(f"CDP Session started for {self.id}")
        
        self.cdp.on("Page.screencastFrame", self._on_frame)
        
        await self.cdp.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": Config.JPEG_QUALITY,
            "maxWidth": Config.WIDTH,
            "maxHeight": Config.HEIGHT,
            "everyNthFrame": 1
        })
        logger.info(f"Screencast started for {self.id}")
        
        await self.page.goto("https://www.google.com")
        self.is_active = True
        
        # Start the High-Speed Frame Engine (Hybrid)
        asyncio.create_task(self._frame_engine())
        logger.info(f"Browser {self.id} ready.")

    async def _frame_engine(self):
        """High-speed hybrid frame engine: CDP + Screenshot Fallback"""
        last_frame_count = 0
        while self.is_active:
            await asyncio.sleep(0.1) # 10fps minimum guarantee
            if self._frames_sent == last_frame_count:
                # CDP is stalled or not sending, force a screenshot
                try:
                    screenshot = await self.page.screenshot(type="jpeg", quality=Config.JPEG_QUALITY, scale="css")
                    b64_data = base64.b64encode(screenshot).decode('utf-8')
                    await self._distribute_frame(b64_data)
                except: pass
            last_frame_count = self._frames_sent

    async def _distribute_frame(self, b64_data: str):
        self._frames_sent += 1
        now = time.time()
        if now - self._last_frame_log > 5:
            logger.info(f"NODE FLOW [{self.id}]: {self._frames_sent} frames delivered in last 5s")
            self._frames_sent = 0
            self._last_frame_log = now

        if self.websockets:
            msg = json.dumps({"type": "frame", "data": b64_data})
            for ws in list(self.websockets):
                try: await ws.send_text(msg)
                except: self.websockets.remove(ws)
        
        if self.tracks:
            img_executor.submit(self._process_and_push_webrtc, b64_data)

    async def _on_frame(self, data):
        if not self.is_active: return
        try:
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": data["sessionId"]})
            await self._distribute_frame(data["data"])
        except: pass

    def _process_and_push_webrtc(self, b64_str: str):
        try:
            raw_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None: return
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = VideoFrame.from_ndarray(img_rgb, format='rgb24')
            frame.pts = self._last_pts
            frame.time_base = fractions.Fraction(1, 90000)
            self._last_pts += 3000
            for track in list(self.tracks):
                track.push_frame(frame)
        except: pass

    async def handle_input(self, action: dict):
        if not self.is_active: return
        try:
            atype = action.get("type")
            x, y = action.get("x", 0), action.get("y", 0)
            logger.info(f"INPUT RECEIVED [{self.id}]: {atype} at ({x}, {y})")
            
            if x is not None: x = max(0, min(int(x), Config.WIDTH))
            if y is not None: y = max(0, min(int(y), Config.HEIGHT))
            
            button = action.get("button", "left")
            
            if atype == "mousemove":
                await self.page.mouse.move(x, y)
            elif atype == "mousedown":
                await self.page.mouse.down(button=button)
            elif atype == "mouseup":
                await self.page.mouse.up(button=button)
            elif atype == "click":
                await self.page.mouse.click(x, y, button=button)
            elif atype == "key":
                key = action.get("key")
                if key:
                    logger.info(f"Keypress {self.id}: {key}")
                    await self.page.keyboard.press(key)
            elif atype == "scroll":
                await self.page.mouse.wheel(0, int(action.get("deltaY", 0)))
            elif atype == "navigate":
                url = action["url"]
                if not url.startswith("http"): url = "https://" + url
                logger.info(f"Navigating {self.id} to {url}")
                await self.page.goto(url)
            elif atype == "refresh": await self.page.reload()
            elif atype == "back": await self.page.go_back()
            elif atype == "forward": await self.page.go_forward()
        except Exception as e:
            logger.error(f"Input Error {self.id}: {e}")

    async def cleanup(self):
        self.is_active = False
        logger.info(f"Cleaning up browser {self.id}")
        try:
            if hasattr(self, 'cdp'): await self.cdp.detach()
            if self.page: await self.page.close()
            if self.context: await self.context.close()
            if self.browser: await self.browser.close()
            if self.playwright: await self.playwright.stop()
        except: pass

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
browsers: Dict[str, BrowserInstance] = {}

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(heartbeat_loop())

async def heartbeat_loop():
    node_url = f"http://{Config.NODE_HOST}:{Config.NODE_PORT}"
    while True:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                await client.post(
                    f"{Config.SERVER_URL}/register",
                    json={
                        "node_id": Config.NODE_ID,
                        "url": node_url,
                        "browsers_count": len(browsers),
                        "browsers": list(browsers.keys())
                    }
                )
        except: pass
        await asyncio.sleep(Config.HEARTBEAT_INTERVAL)

@app.post("/create")
async def create_browser():
    if len(browsers) >= Config.MAX_BROWSERS:
        return {"status": "error", "message": "Max browsers reached"}
    bid = str(uuid.uuid4())[:8]
    instance = BrowserInstance(bid)
    await instance.launch()
    browsers[bid] = instance
    return {"id": bid}

@app.post("/close/{bid}")
async def close_browser(bid: str):
    if bid in browsers:
        await browsers[bid].cleanup()
        del browsers[bid]
    return {"status": "ok"}

@app.post("/api/offer/{bid}")
async def webrtc_offer(bid: str, request: Request):
    if bid not in browsers: return {"error": "not_found"}
    logger.info(f"WebRTC Offer for {bid}")
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    try:
        pc = RTCPeerConnection(configuration=RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        ))
        browsers[bid].pcs.append(pc)
        
        track = BrowserVideoTrack()
        browsers[bid].tracks.append(track)
        
        # Manually create transceiver with explicit direction to avoid aiortc bugs
        pc.addTransceiver("video", direction="sendonly")
        sender = pc.getSenders()[0]
        await sender.replaceTrack(track)
        
        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"DataChannel created for {bid}")
            @channel.on("message")
            def on_message(message):
                try:
                    action = json.loads(message)
                    asyncio.create_task(browsers[bid].handle_input(action))
                except Exception as e:
                    logger.error(f"DC Parse Error: {e}")

        @pc.on("connectionstatechange")
        async def on_state():
            logger.info(f"PC State {bid}: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                if pc in browsers[bid].pcs: browsers[bid].pcs.remove(pc)
                if track in browsers[bid].tracks: browsers[bid].tracks.remove(track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    except Exception as e:
        logger.error(f"WebRTC Handshake Failed: {e}")
        # Cleanup
        if pc in browsers[bid].pcs: browsers[bid].pcs.remove(pc)
        return {"error": str(e)}

@app.websocket("/ws/{bid}")
async def browser_ws(websocket: WebSocket, bid: str):
    if bid not in browsers:
        await websocket.close()
        return
    await websocket.accept()
    instance = browsers[bid]
    instance.websockets.append(websocket)
    logger.info(f"WS Connected {bid}")
    try:
        while True:
            data = await websocket.receive_text()
            action = json.loads(data)
            asyncio.create_task(instance.handle_input(action))
    except WebSocketDisconnect:
        logger.warning(f"WS Disconnected {bid}")
    except Exception as e:
        logger.error(f"WS Error {bid}: {e}")
    finally:
        if websocket in instance.websockets:
            instance.websockets.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=Config.NODE_PORT, log_level="info")
