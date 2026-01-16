import os
import uuid
import time
import json
import asyncio
import base64
import platform
import httpx
import io
from PIL import Image
from typing import Dict, Any, Optional, List

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer
from av import VideoFrame
import fractions
from fastapi.middleware.cors import CORSMiddleware

# Stealth/Undetected imports
try:
    import undetected_playwright as undpw
    ud_async_playwright = getattr(undpw, "async_playwright", None)
except Exception:
    ud_async_playwright = None

# ======================================================================
# CONFIG
# ======================================================================

class Config:
    NODE_ID = str(uuid.uuid4())[:8]
    SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
    NODE_HOST = os.getenv("NODE_HOST", "localhost")
    NODE_PORT = int(os.getenv("NODE_PORT", 8001))
    HEADLESS = True
    VIEWPORT = {"width": 1280, "height": 720}
    FPS = 15
    IS_WINDOWS = platform.system() == "Windows"

# ======================================================================
# STEALTH SCRIPT
# ======================================================================

STEALTH_JS = r"""
(function() {
    // 1. Hide webdriver
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    
    // 2. Languages
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    
    // 3. Fake Plugins
    const mockPlugins = [
        { name: 'Chrome PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
        { name: 'Microsoft Edge PDF Viewer', filename: 'internal-pdf-viewer', description: 'Portable Document Format' }
    ];
    Object.defineProperty(navigator, 'plugins', { get: () => mockPlugins });

    // 4. Chrome object
    window.chrome = { runtime: {}, app: {}, csi: () => {}, loadTimes: () => {} };

    // 5. WebGL
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(p) {
        if (p === 37445) return 'Google Inc. (Intel)';
        if (p === 37446) return 'ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)';
        return getParameter.apply(this, arguments);
    };

    // 6. Canvas Noise (Anti-Fingerprinting)
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(t, a) {
        const c = originalGetContext.call(this, t, a);
        if (t === '2d' && c) {
            const org = c.getImageData;
            c.getImageData = function(x, y, w, h) {
                const d = org.call(this, x, y, w, h);
                // Subtle noise to the first pixel
                d.data[0] = d.data[0] + (Math.random() > 0.5 ? 1 : -1);
                return d;
            };
        }
        return c;
    };

    // 7. Proxy protection
    const makeProxy = (fn) => new Proxy(fn, { 
        get: (t, p) => p === 'toString' ? () => 'function () { [native code] }' : t[p] 
    });
    window.navigator.permissions.query = makeProxy(window.navigator.permissions.query);
})();
"""

# ======================================================================
# BROWSER INSTANCE
# ======================================================================

class BrowserVideoTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue(maxsize=2)

    async def recv(self):
        frame = await self.queue.get()
        return frame

class BrowserInstance:
    def __init__(self, browser_id: str):
        self.id = browser_id
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.active_websockets: List[WebSocket] = []
        self.peer_connections = set()
        self.active_tracks = set() # Set[BrowserVideoTrack]
        self.is_running = False
        self.cdp = None
        self._timestamp = 0

    async def start(self):
        if ud_async_playwright:
            self.playwright = await ud_async_playwright().start()
        else:
            self.playwright = await async_playwright().start()

        args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--use-gl=desktop",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--disable-infobars",
            "--window-position=0,0",
            "--disable-canvas-aa", # Performance
            "--disable-2d-canvas-clip-utils", # Performance
            "--disable-gl-drawing-for-tests",
        ]
        
        if Config.IS_WINDOWS and Config.HEADLESS:
            args.append("--disable-gpu")

        self.browser = await self.playwright.chromium.launch(
            headless=Config.HEADLESS,
            args=args,
            channel="chrome"
        )

        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        self.context = await self.browser.new_context(
            viewport=Config.VIEWPORT, 
            user_agent=ua,
            device_scale_factor=1,
        )
        
        await self.context.add_init_script(STEALTH_JS)
        self.page = await self.context.new_page()
        await self.page.add_init_script("document.documentElement.style.background = '#000';")
        
        await self.page.goto("https://www.google.com", wait_until="networkidle")
        self.is_running = True
        
        # Start High-Performance CDP Screencast
        self.cdp = await self.page.context.new_cdp_session(self.page)
        await self.cdp.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": 40, # Lower quality for much higher speed
            "maxWidth": 1280,
            "maxHeight": 720,
            "everyNthFrame": 1
        })
        self.cdp.on("Page.screencastFrame", self._on_screencast_frame)

    async def _on_screencast_frame(self, data):
        """Callback for CDP screencast frames"""
        try:
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": data["sessionId"]})
            b64_data = data["data"]
            image_bytes = base64.b64decode(b64_data)
            
            # 1. Prepare frame for WebRTC
            if self.active_tracks:
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    new_frame = VideoFrame.from_image(img)
                    new_frame.pts = self._timestamp
                    new_frame.time_base = fractions.Fraction(1, 90000)
                    self._timestamp += 3000

                    # Push to all active WebRTC tracks
                    for track in list(self.active_tracks):
                        if track.queue.full():
                            try: track.queue.get_nowait()
                            except: pass
                        track.queue.put_nowait(new_frame)
                except Exception as e:
                    print(f"Frame encoding error: {e}")
            
            # 2. Update WebSockets
            if self.active_websockets:
                msg = json.dumps({"type": "frame", "data": b64_data})
                for ws in list(self.active_websockets):
                    try: await ws.send_text(msg)
                    except: 
                        if ws in self.active_websockets: self.active_websockets.remove(ws)
        except Exception as e:
            print(f"Screencast processing error: {e}")

    async def stop(self):
        self.is_running = False
        try:
            if hasattr(self, 'cdp'): await self.cdp.detach()
        except: pass
        if self.page: await self.page.close()
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()

# ======================================================================
# NODE APP
# ======================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

browsers: Dict[str, BrowserInstance] = {}

async def register_with_server():
    node_url = f"http://{Config.NODE_HOST}:{Config.NODE_PORT}"
    while True:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
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
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup():
    print(f"Node {Config.NODE_ID} UP on port {Config.NODE_PORT}")
    asyncio.create_task(register_with_server())

@app.post("/create")
async def create_browser():
    browser_id = str(uuid.uuid4())[:8]
    instance = BrowserInstance(browser_id)
    await instance.start()
    browsers[browser_id] = instance
    return {"id": browser_id}

@app.post("/close/{browser_id}")
async def close_browser(browser_id: str):
    if browser_id in browsers:
        await browsers[browser_id].stop()
        del browsers[browser_id]
        return {"status": "closed"}
    return {"status": "not_found"}

@app.post("/close_all")
async def close_all_browsers():
    ids = list(browsers.keys())
    for bid in ids:
        await browsers[bid].stop()
        del browsers[bid]
    return {"status": "all_closed"}

@app.post("/api/offer/{browser_id}")
async def webrtc_offer(browser_id: str, request: Request):
    print(f"Received WebRTC offer for {browser_id}")
    if browser_id not in browsers:
        return {"error": "not_found"}
    
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    pc = RTCPeerConnection()
    instance = browsers[browser_id]
    instance.peer_connections.add(pc)
    
    # Create per-session track
    track = BrowserVideoTrack()
    instance.active_tracks.add(track)
    pc.addTrack(track)
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            instance.peer_connections.discard(pc)
            instance.active_tracks.discard(track)
    
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

@app.websocket("/ws/{browser_id}")
async def websocket_endpoint(websocket: WebSocket, browser_id: str):
    if browser_id not in browsers:
        await websocket.close()
        return
    await websocket.accept()
    instance = browsers[browser_id]
    instance.active_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            action = json.loads(data)
            type = action.get("type")
            
            if type == "navigate":
                url = action.get("url", "")
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                await instance.page.goto(url)
            
            elif type == "mousemove":
                await instance.page.mouse.move(action["x"], action["y"])
            
            elif type == "mousedown":
                await instance.page.mouse.down()
            
            elif type == "mouseup":
                await instance.page.mouse.up()
            
            elif type == "click":
                await instance.page.mouse.click(action["x"], action["y"])
            
            elif type == "type":
                await instance.page.keyboard.type(action["text"])
            
            elif type == "key":
                await instance.page.keyboard.press(action["key"])
            
            elif type == "scroll":
                await instance.page.mouse.wheel(0, action["deltaY"])
                
    except WebSocketDisconnect:
        if websocket in instance.active_websockets: instance.active_websockets.remove(websocket)
    except Exception as e:
        print(f"WS error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=Config.NODE_PORT)
