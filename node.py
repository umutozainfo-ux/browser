import os
import uuid
import time
import json
import asyncio
import base64
import platform
import httpx
from typing import Dict, Any, Optional, List

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

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

class BrowserInstance:
    def __init__(self, browser_id: str):
        self.id = browser_id
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.active_websockets: List[WebSocket] = []
        self.is_running = False
        self._capture_task = None

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
        ]
        
        # On Windows Headless, GPU can cause blank screens
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
            is_mobile=False,
            has_touch=False,
        )
        
        await self.context.add_init_script(STEALTH_JS)
        self.page = await self.context.new_page()
        
        # Ensure we set a background color just in case
        await self.page.add_init_script("document.documentElement.style.background = '#000';")
        
        await self.page.goto("https://www.google.com", wait_until="networkidle")
        self.is_running = True
        self._capture_task = asyncio.create_task(self._stream_loop())

    async def _stream_loop(self):
        delay = 1.0 / Config.FPS
        while self.is_running:
            try:
                if self.active_websockets:
                    # Capturing with type="jpeg" helps on Windows headless
                    screenshot = await self.page.screenshot(type="jpeg", quality=60, full_page=False)
                    b64_data = base64.b64encode(screenshot).decode('utf-8')
                    msg = json.dumps({"type": "frame", "data": b64_data})
                    
                    for ws in self.active_websockets[:]:
                        try: 
                            await ws.send_text(msg)
                        except: 
                            if ws in self.active_websockets:
                                self.active_websockets.remove(ws)
            except Exception as e:
                # print(f"Capture error: {e}")
                pass
            await asyncio.sleep(delay)

    async def stop(self):
        self.is_running = False
        if self._capture_task: self._capture_task.cancel()
        if self.page: await self.page.close()
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()

# ======================================================================
# NODE APP
# ======================================================================

app = FastAPI()
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
