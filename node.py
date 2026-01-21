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
    // 1. Webdriver evasion
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    
    // 2. WebGL Evasion
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(p) {
        if (p === 37445) return 'Google Inc. (Intel)';
        if (p === 37446) return 'ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)';
        return getParameter.apply(this, arguments);
    };

    // 3. Hardware/Memory spoofing
    Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
    Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });

    // 4. Plugin Mocking
    const mockPlugins = [
        { name: 'PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
        { name: 'Chrome PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' }
    ];
    Object.defineProperty(navigator, 'plugins', { get: () => mockPlugins });

    // 5. Languages & Permissions
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
        Promise.resolve({ state: Notification.permission }) :
        originalQuery(parameters)
    );

    // 6. Chrome Object
    window.chrome = {
        app: { isInstalled: false, InstallState: { DISABLED: 'DISABLED', INSTALLED: 'INSTALLED', NOT_INSTALLED: 'NOT_INSTALLED' }, RunningState: { CANNOT_RUN: 'CANNOT_RUN', READY_TO_RUN: 'READY_TO_RUN', RUNNING: 'RUNNING' } },
        runtime: { OnInstalledReason: { CHROME_UPDATE: 'chrome_update', INSTALL: 'install', SHARED_MODULE_UPDATE: 'shared_module_update', UPDATE: 'update' }, OnRestartRequiredReason: { APP_UPDATE: 'app_update', OS_UPDATE: 'os_update', PERIODIC: 'periodic' }, PlatformArch: { ARM: 'arm', ARM64: 'arm64', MIPS: 'mips', MIPS64: 'mips64', X86_32: 'x86-32', X86_64: 'x86-64' }, PlatformNaclArch: { ARM: 'arm', MIPS: 'mips', MIPS64: 'mips64', X86_32: 'x86-32', X86_64: 'x86-64' }, PlatformOs: { ANDROID: 'android', CROS: 'cros', LINUX: 'linux', MAC: 'mac', OPENBSD: 'openbsd', WIN: 'win' }, RequestUpdateCheckStatus: { NO_UPDATE: 'no_update', THROTTLED: 'throttled', UPDATE_AVAILABLE: 'update_available' } },
        loadTimes: () => ({
            requestTime: Date.now() / 1000 - 0.5,
            startLoadTime: Date.now() / 1000 - 0.5,
            commitLoadTime: Date.now() / 1000 - 0.4,
            finishDocumentLoadTime: Date.now() / 1000 - 0.3,
            finishLoadTime: Date.now() / 1000 - 0.2,
            firstPaintTime: Date.now() / 1000 - 0.35,
            firstPaintAfterLoadTime: 0,
            navigationType: 'Other',
            wasFetchedViaSpdy: true,
            wasNpnNegotiated: true,
            wasAlternateProtocolAvailable: false,
            connectionInfo: 'h2'
        }),
        csi: () => ({ startE: Date.now() - 500, onloadT: Date.now(), pageT: 500.2, tran: 15 })
    };

    // 7. Mouse Event Humanization (Prevent detection of synthetic events)
    const originalDispatch = Element.prototype.dispatchEvent;
    Element.prototype.dispatchEvent = function(event) {
        if (event instanceof MouseEvent && !event.isTrusted) {
            // Some sites check isTrusted, which we can't easily spoof here, 
            // but we can make sure the event properties look good.
        }
        return originalDispatch.apply(this, arguments);
    };

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
        self.input_lock = asyncio.Lock()

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
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            java_script_enabled=True,
            bypass_csp=True,
            ignore_https_errors=True,
            color_scheme='dark',
            locale='en-US',
            timezone_id='America/New_York'
        )
        
        await self.context.add_init_script(STEALTH_JS)
        self.page = await self.context.new_page()
        
        self.cdp = await self.context.new_cdp_session(self.page)
        logger.info(f"CDP Session started for {self.id}")
        
        self.cdp.on("Page.screencastFrame", self._on_frame)
        
        # Tracking for mouse state
        self._mouse_x = 0
        self._mouse_y = 0
        self._mouse_down = False
        
        await self.cdp.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": 80, # Improved quality for CAPTCHA solving
            "maxWidth": Config.WIDTH,
            "maxHeight": Config.HEIGHT,
            "everyNthFrame": 1
        })
        logger.info(f"Screencast started for {self.id}")
        
        await self.page.goto("https://www.tiktok.com") # Changed default for tests
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

    def _bezier_curve(self, start_x, start_y, end_x, end_y, steps=20):
        """Generate smooth Bézier curve points for human-like mouse movement"""
        # Create random control points for natural curve
        ctrl1_x = start_x + np.random.uniform(-50, 50) + (end_x - start_x) * 0.3
        ctrl1_y = start_y + np.random.uniform(-50, 50) + (end_y - start_y) * 0.3
        ctrl2_x = start_x + np.random.uniform(-50, 50) + (end_x - start_x) * 0.7
        ctrl2_y = start_y + np.random.uniform(-50, 50) + (end_y - start_y) * 0.7
        
        points = []
        for i in range(steps + 1):
            t = i / steps
            # Cubic Bézier formula
            x = (1-t)**3 * start_x + 3*(1-t)**2*t * ctrl1_x + 3*(1-t)*t**2 * ctrl2_x + t**3 * end_x
            y = (1-t)**3 * start_y + 3*(1-t)**2*t * ctrl1_y + 3*(1-t)*t**2 * ctrl2_y + t**3 * end_y
            
            # Add micro jitter for realism
            jitter_x = np.random.uniform(-1.5, 1.5)
            jitter_y = np.random.uniform(-1.5, 1.5)
            
            points.append((
                max(0, min(int(x + jitter_x), Config.WIDTH)),
                max(0, min(int(y + jitter_y), Config.HEIGHT))
            ))
        
        return points
    
    async def _human_move(self, from_x, from_y, to_x, to_y, is_dragging=False):
        """Move mouse in a human-like way with realistic timing and path"""
        distance = np.sqrt((to_x - from_x)**2 + (to_y - from_y)**2)
        
        # Determine how many steps based on distance and whether dragging
        if distance < 10:
            # Very short distance: just go direct
            await self.page.mouse.move(to_x, to_y)
            return
        
        # Calculate adaptive steps based on distance
        if is_dragging:
            # During drag (CAPTCHA): more steps for smoothness + realism
            steps = max(15, min(35, int(distance / 10)))
        else:
            # Normal movement: fewer steps
            steps = max(5, min(20, int(distance / 20)))
        
        # Generate Bézier curve path
        path = self._bezier_curve(from_x, from_y, to_x, to_y, steps)
        
        # Move along path with realistic variable timing
        for i, (px, py) in enumerate(path):
            # Calculate delay with acceleration/deceleration
            progress = i / len(path)
            
            if is_dragging:
                # CAPTCHA drag: start slow, speed up, slow down at end (human overshoot correction)
                if progress < 0.1:
                    # Acceleration phase
                    delay = np.random.uniform(0.008, 0.015)
                elif progress > 0.85:
                    # Deceleration + micro-adjustments
                    delay = np.random.uniform(0.010, 0.020)
                else:
                    # Steady movement with slight variation
                    delay = np.random.uniform(0.003, 0.008)
            else:
                # Normal movement: faster consistent motion
                delay = np.random.uniform(0.002, 0.006)
            
            await self.page.mouse.move(px, py)
            await asyncio.sleep(delay)
        
        # Final micro-adjustment for CAPTCHA precision
        if is_dragging and distance > 50:
            # Add subtle overcorrection then settle (human behavior)
            overshoot_x = to_x + np.random.randint(-2, 3)
            overshoot_y = to_y + np.random.randint(-2, 3)
            await self.page.mouse.move(overshoot_x, overshoot_y)
            await asyncio.sleep(np.random.uniform(0.015, 0.030))
            await self.page.mouse.move(to_x, to_y)
            await asyncio.sleep(np.random.uniform(0.005, 0.015))

    async def handle_input(self, action: dict):
        if not self.is_active: return
        async with self.input_lock: # Ensure sequential processing
            try:
                atype = action.get("type")
                x, y = action.get("x", 0), action.get("y", 0)
                
                if x is not None: x = max(0, min(int(x), Config.WIDTH))
                if y is not None: y = max(0, min(int(y), Config.HEIGHT))
                
                button = action.get("button", "left")
                
                # Tracking for interpolation
                prev_x, prev_y = self._mouse_x, self._mouse_y
                self._mouse_x, self._mouse_y = x, y

                if atype == "mousemove":
                    # DIRECT PASS-THROUGH FOR INTERACTIVE CONTROL
                    # When a user moves the mouse interactively, we must NOT add artificial delays or curves.
                    # The user's own hand provides the human path and timing.
                    # Adding sleeps here causes lag, breaking drag-and-drop operations.
                    if self._mouse_down:
                        await self.page.mouse.move(x, y, steps=1) # Instant update for drag
                    else:
                        # For hover, we can afford slightly more smoothing if needed, but direct is best for responsiveness
                        await self.page.mouse.move(x, y, steps=1)
                
                elif atype == "mousedown":
                    # Approach the target human-like if we aren't there yet
                    if abs(x - prev_x) > 5 or abs(y - prev_y) > 5:
                        await self._human_move(prev_x, prev_y, x, y, is_dragging=False)
                    
                    # Human-like click timing
                    await asyncio.sleep(np.random.uniform(0.020, 0.050))
                    self._mouse_down = True
                    await self.page.mouse.down(button=button)
                    logger.debug(f"MOUSE DOWN [{self.id}]: ({x}, {y})")
                
                elif atype == "mouseup":
                    # Release with slight natural delay
                    await asyncio.sleep(np.random.uniform(0.015, 0.040))
                    self._mouse_down = False
                    await self.page.mouse.up(button=button)
                    logger.debug(f"MOUSE UP [{self.id}]")

                elif atype == "click":
                    # Full automated click sequence (Approach -> Down -> Up)
                    if abs(x - prev_x) > 5 or abs(y - prev_y) > 5:
                        await self._human_move(prev_x, prev_y, x, y, is_dragging=False)
                    
                    await asyncio.sleep(np.random.uniform(0.030, 0.070))
                    await self.page.mouse.down(button=button)
                    await asyncio.sleep(np.random.uniform(0.050, 0.120))
                    await self.page.mouse.up(button=button)
                    await asyncio.sleep(np.random.uniform(0.010, 0.030))
                
                elif atype == "key":
                    key = action.get("key")
                    if key:
                        await self.page.keyboard.down(key)
                        await asyncio.sleep(np.random.uniform(0.030, 0.080))
                        await self.page.keyboard.up(key)
                
                elif atype == "scroll":
                    await self.page.mouse.wheel(0, int(action.get("deltaY", 0)))
                
                elif atype == "navigate":
                    url = action["url"]
                    if not url.startswith("http"): url = "https://" + url
                    await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
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
