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

class FingerprintManager:
    """Generates unique browser fingerprints for evasion and multi-accounting."""
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
    ]

    @staticmethod
    def generate():
        import random
        # Randomize hardware specs
        cores = random.choice([4, 8, 12, 16])
        memory = random.choice([4, 8, 16, 32])
        
        # Randomize viewport slightly
        width = 1280 + random.randint(-20, 20)
        height = 720 + random.randint(-20, 20)
        
        # Randomize Locales/Timezones
        locales = ["en-US", "en-GB", "fr-FR", "de-DE", "es-ES"]
        timezones = ["America/New_York", "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Madrid"]
        idx = random.randint(0, len(locales)-1)
        
        return {
            "user_agent": random.choice(FingerprintManager.USER_AGENTS),
            "viewport": {"width": width, "height": height},
            "device_memory": memory,
            "hardware_concurrency": cores,
            "locale": locales[idx],
            "timezone_id": timezones[idx],
            "webgl_vendor": random.choice(["Google Inc. (Intel)", "Google Inc. (NVIDIA)", "Google Inc. (AMD)"]),
            "webgl_renderer": random.choice([
                "ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)",
                "ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0, D3D11)",
                "ANGLE (AMD, Radeon RX 6600 Direct3D11 vs_5_0 ps_5_0, D3D11)"
            ])
        }

class Config:
    NODE_ID = os.getenv("NODE_ID", f"node-{str(uuid.uuid4())[:6]}")
    SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")
    NODE_HOST = os.getenv("NODE_HOST", "localhost")
    NODE_PORT = int(os.getenv("NODE_PORT", 8002))
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
    // ============================================
    // 1. AGGRESSIVE WEBDRIVER EVASION
    // ============================================
    // The key insight: navigator.webdriver should either:
    // - Not exist at all (no property)
    // - Return false (real Chrome when not automated)
    // We'll make it return false AND hide it from property enumeration
    
    // First, delete from prototype to remove the original
    try {
        delete Object.getPrototypeOf(navigator).webdriver;
    } catch (e) {}
    
    // Define as false (real Chrome returns false when not automated)
    Object.defineProperty(navigator, 'webdriver', {
        get: () => false,
        configurable: true,
        enumerable: false  // Hide from enumeration
    });
    
    // Intercept hasOwnProperty checks
    const originalHasOwnProperty = Object.prototype.hasOwnProperty;
    Object.prototype.hasOwnProperty = function(prop) {
        if (prop === 'webdriver' && this === navigator) {
            return false;  // Pretend it doesn't exist as own property
        }
        return originalHasOwnProperty.call(this, prop);
    };
    
    // Spoof the property descriptor to look natural
    const originalGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
    Object.getOwnPropertyDescriptor = function(obj, prop) {
        if (prop === 'webdriver' && obj === navigator) {
            return undefined;  // Hide from descriptor checks
        }
        return originalGetOwnPropertyDescriptor.apply(this, arguments);
    };
    
    // Intercept property name enumeration
    const originalGetOwnPropertyNames = Object.getOwnPropertyNames;
    Object.getOwnPropertyNames = function(obj) {
        const props = originalGetOwnPropertyNames.apply(this, arguments);
        if (obj === navigator) {
            return props.filter(p => p !== 'webdriver');
        }
        return props;
    };
    
    // Also intercept Object.keys
    const originalKeys = Object.keys;
    Object.keys = function(obj) {
        const keys = originalKeys.apply(this, arguments);
        if (obj === navigator) {
            return keys.filter(k => k !== 'webdriver');
        }
        return keys;
    };

    // ============================================
    // 2. WebGL Evasion
    // ============================================
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(p) {
        if (p === 37445) return 'Google Inc. (Intel)';
        if (p === 37446) return 'ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)';
        return getParameter.apply(this, arguments);
    };
    
    // Also handle WebGL2
    if (typeof WebGL2RenderingContext !== 'undefined') {
        const getParameter2 = WebGL2RenderingContext.prototype.getParameter;
        WebGL2RenderingContext.prototype.getParameter = function(p) {
            if (p === 37445) return 'Google Inc. (Intel)';
            if (p === 37446) return 'ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)';
            return getParameter2.apply(this, arguments);
        };
    }

    // ============================================
    // 3. Hardware/Memory spoofing
    // ============================================
    Object.defineProperty(navigator, 'deviceMemory', { get: () => 8, configurable: true });
    Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8, configurable: true });

    // ============================================
    // 4. PROPER PluginArray MOCKING
    // ============================================
    // Create a proper PluginArray-like structure that passes instanceof checks
    (function mockPlugins() {
        // Create mock Plugin objects
        function createPlugin(name, description, filename, mimeTypes) {
            const plugin = Object.create(Plugin.prototype);
            Object.defineProperties(plugin, {
                name: { value: name, enumerable: true },
                description: { value: description, enumerable: true },
                filename: { value: filename, enumerable: true },
                length: { value: mimeTypes.length, enumerable: true }
            });
            mimeTypes.forEach((mt, i) => {
                const mimeType = Object.create(MimeType.prototype);
                Object.defineProperties(mimeType, {
                    type: { value: mt.type, enumerable: true },
                    suffixes: { value: mt.suffixes, enumerable: true },
                    description: { value: mt.description, enumerable: true },
                    enabledPlugin: { value: plugin, enumerable: true }
                });
                Object.defineProperty(plugin, i, { value: mimeType, enumerable: true });
            });
            return plugin;
        }

        const pdfPlugin = createPlugin(
            'PDF Viewer',
            'Portable Document Format',
            'internal-pdf-viewer',
            [{ type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format' }]
        );
        
        const chromePdfPlugin = createPlugin(
            'Chrome PDF Viewer',
            'Portable Document Format',
            'internal-pdf-viewer',
            [{ type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format' }]
        );
        
        const chromiumPdfPlugin = createPlugin(
            'Chromium PDF Viewer',
            'Portable Document Format',
            'internal-pdf-viewer',
            [{ type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format' }]
        );

        // Create a proper PluginArray that inherits from PluginArray.prototype
        const pluginsArray = Object.create(PluginArray.prototype);
        const plugins = [pdfPlugin, chromePdfPlugin, chromiumPdfPlugin];
        
        plugins.forEach((plugin, i) => {
            Object.defineProperty(pluginsArray, i, { value: plugin, enumerable: true });
            Object.defineProperty(pluginsArray, plugin.name, { value: plugin, enumerable: false });
        });
        
        Object.defineProperty(pluginsArray, 'length', { value: plugins.length, enumerable: true });
        
        // Add item() and namedItem() methods
        pluginsArray.item = function(index) { return plugins[index] || null; };
        pluginsArray.namedItem = function(name) { return plugins.find(p => p.name === name) || null; };
        pluginsArray.refresh = function() {};
        
        // Make it iterable
        pluginsArray[Symbol.iterator] = function* () {
            for (let i = 0; i < plugins.length; i++) yield plugins[i];
        };

        Object.defineProperty(navigator, 'plugins', {
            get: () => pluginsArray,
            configurable: true
        });
        
        // Also mock mimeTypes
        const mimeTypesArray = Object.create(MimeTypeArray.prototype);
        const mimeTypes = [
            { type: 'application/pdf', suffixes: 'pdf', description: 'Portable Document Format', plugin: pdfPlugin }
        ];
        
        mimeTypes.forEach((mt, i) => {
            const mimeType = Object.create(MimeType.prototype);
            Object.defineProperties(mimeType, {
                type: { value: mt.type, enumerable: true },
                suffixes: { value: mt.suffixes, enumerable: true },
                description: { value: mt.description, enumerable: true },
                enabledPlugin: { value: mt.plugin, enumerable: true }
            });
            Object.defineProperty(mimeTypesArray, i, { value: mimeType, enumerable: true });
            Object.defineProperty(mimeTypesArray, mt.type, { value: mimeType, enumerable: false });
        });
        
        Object.defineProperty(mimeTypesArray, 'length', { value: mimeTypes.length, enumerable: true });
        mimeTypesArray.item = function(index) { return mimeTypes[index] || null; };
        mimeTypesArray.namedItem = function(name) { return mimeTypes.find(m => m.type === name) || null; };
        mimeTypesArray[Symbol.iterator] = function* () {
            for (let i = 0; i < mimeTypes.length; i++) yield mimeTypes[i];
        };
        
        Object.defineProperty(navigator, 'mimeTypes', {
            get: () => mimeTypesArray,
            configurable: true
        });
    })();

    // ============================================
    // 5. Languages & Permissions
    // ============================================
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'], configurable: true });
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
        Promise.resolve({ state: Notification.permission }) :
        originalQuery(parameters)
    );

    // ============================================
    // 6. Chrome Object (fully mock the runtime)
    // ============================================
    window.chrome = {
        app: { isInstalled: false, InstallState: { DISABLED: 'DISABLED', INSTALLED: 'INSTALLED', NOT_INSTALLED: 'NOT_INSTALLED' }, RunningState: { CANNOT_RUN: 'CANNOT_RUN', READY_TO_RUN: 'READY_TO_RUN', RUNNING: 'RUNNING' } },
        runtime: { 
            OnInstalledReason: { CHROME_UPDATE: 'chrome_update', INSTALL: 'install', SHARED_MODULE_UPDATE: 'shared_module_update', UPDATE: 'update' }, 
            OnRestartRequiredReason: { APP_UPDATE: 'app_update', OS_UPDATE: 'os_update', PERIODIC: 'periodic' }, 
            PlatformArch: { ARM: 'arm', ARM64: 'arm64', MIPS: 'mips', MIPS64: 'mips64', X86_32: 'x86-32', X86_64: 'x86-64' }, 
            PlatformNaclArch: { ARM: 'arm', MIPS: 'mips', MIPS64: 'mips64', X86_32: 'x86-32', X86_64: 'x86-64' }, 
            PlatformOs: { ANDROID: 'android', CROS: 'cros', LINUX: 'linux', MAC: 'mac', OPENBSD: 'openbsd', WIN: 'win' }, 
            RequestUpdateCheckStatus: { NO_UPDATE: 'no_update', THROTTLED: 'throttled', UPDATE_AVAILABLE: 'update_available' },
            connect: function() {},
            sendMessage: function() {}
        },
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

    // ============================================
    // 7. Mouse Event Humanization
    // ============================================
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
    
    // ============================================
    // 8. Additional Anti-Detection Measures
    // ============================================
    // Spoof the toString of overridden functions to look native
    const nativeCode = 'function () { [native code] }';
    const spoofFunctionToString = (fn, name) => {
        const handler = {
            apply: function(target, thisArg, args) {
                if (thisArg === fn) {
                    return `function ${name || ''}() { [native code] }`;
                }
                return target.apply(thisArg, args);
            }
        };
        Function.prototype.toString = new Proxy(Function.prototype.toString, handler);
    };
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
    def __init__(self, bid: str, mode: str = "ephemeral", profile_id: Optional[str] = None, owner: str = "system"):
        self.id = bid
        self.mode = mode
        self.owner = owner
        self.profile_id = profile_id or f"prof_{bid}"
        self.fingerprint = FingerprintManager.generate()
        
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
        
        # Path for persistent profiles
        self.profile_path = os.path.abspath(f"profiles/{self.profile_id}")
        if self.mode == "persistent":
             os.makedirs(self.profile_path, exist_ok=True)
             logger.info(f"Initialized persistent profile at {self.profile_path}")

    async def launch(self):
        logger.info(f"Launching {self.mode} browser {self.id} with profile {self.profile_id}...")
        self.playwright = await async_playwright().start()
        
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--window-position=0,0",
            "--disable-infobars",
            "--mute-audio",
            "--disable-canvas-aa",
            "--disable-2d-canvas-clip-utils",
            "--use-gl=angle" if platform.system() == "Windows" else "--use-gl=desktop",
            f"--window-size={self.fingerprint['viewport']['width']},{self.fingerprint['viewport']['height']}",
            # Additional anti-detection flags
            "--disable-automation",
            "--disable-extensions",
            "--disable-default-apps",
            "--disable-component-extensions-with-background-pages",
            "--disable-background-networking",
            "--disable-sync",
            "--metrics-recording-only",
            "--disable-hang-monitor",
            "--disable-prompt-on-repost",
            "--no-first-run",
            "--enable-features=NetworkService,NetworkServiceInProcess,Vulkan",
            "--flag-switches-begin",
            "--flag-switches-end",
            # Performance flags
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-breakpad",
            "--disable-component-update",
            "--disable-domain-reliability",
            "--disable-features=AudioServiceOutOfProcess",
            "--disable-ipc-flooding-protection",
            "--disable-renderer-backgrounding",
            "--enable-automation", # Keep for some features but hide via JS
            "--force-color-profile=srgb",
            "--js-flags=--max-old-space-size=4096",
        ]

        context_params = {
            "viewport": self.fingerprint["viewport"],
            "user_agent": self.fingerprint["user_agent"],
            "java_script_enabled": True,
            "bypass_csp": True,
            "ignore_https_errors": True,
            "color_scheme": 'dark',
            "locale": self.fingerprint["locale"],
            "timezone_id": self.fingerprint["timezone_id"],
        }

        try:
            if self.mode == "persistent":
                # Ensure path is absolutely clean for Windows
                self.profile_path = os.path.normpath(self.profile_path)
                os.makedirs(self.profile_path, exist_ok=True)
                
                # Persistent mode: Stores cookies, sessions, etc.
                self.context = await self.playwright.chromium.launch_persistent_context(
                    user_data_dir=self.profile_path,
                    headless=Config.HEADLESS,
                    args=launch_args,
                    channel="chrome",
                    handle_sigint=False,
                    handle_sigterm=False,
                    handle_sighup=False,
                    **context_params
                )
                if self.context.pages:
                    self.page = self.context.pages[0]
                else:
                    self.page = await self.context.new_page()
            else:
                # Ephemeral mode: Fresh start, no data saved
                self.browser = await self.playwright.chromium.launch(
                    headless=Config.HEADLESS,
                    args=launch_args,
                    channel="chrome"
                )
                self.context = await self.browser.new_context(**context_params)
                self.page = await self.context.new_page()
        except Exception as e:
            logger.error(f"Launch failed for {self.id}: {str(e)}")
            raise e
        
        # Dynamic Stealth Injection with Fingerprint
        stealth_script = STEALTH_JS.replace("'Google Inc. (Intel)'", f"'{self.fingerprint['webgl_vendor']}'")
        stealth_script = stealth_script.replace("'ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)'", f"'{self.fingerprint['webgl_renderer']}'")
        # Add hardware concurrency and memory to JS injection
        stealth_script = stealth_script.replace("deviceMemory', { get: () => 8 }", f"deviceMemory', {{ get: () => {self.fingerprint['device_memory']} }}")
        stealth_script = stealth_script.replace("hardwareConcurrency', { get: () => 8 }", f"hardwareConcurrency', {{ get: () => {self.fingerprint['hardware_concurrency']} }}")

        await self.context.add_init_script(stealth_script)
        
        self.cdp = await self.context.new_cdp_session(self.page)
        logger.info(f"CDP Session started for {self.id}")
        
        # ============================================
        # CRITICAL: CDP-LEVEL WEBDRIVER EVASION
        # ============================================
        # This runs BEFORE any page scripts, ensuring webdriver is hidden
        # from even the earliest detection attempts
        webdriver_evasion_script = """
        (() => {
            // Delete webdriver from Navigator prototype chain
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
                configurable: true
            });
            
            // Also try to delete it from the prototype
            try {
                const proto = Object.getPrototypeOf(navigator);
                if (proto.hasOwnProperty('webdriver')) {
                    delete proto.webdriver;
                }
            } catch(e) {}
            
            // Intercept any future attempts to check webdriver
            const originalHasOwnProperty = Object.prototype.hasOwnProperty;
            Object.prototype.hasOwnProperty = function(prop) {
                if (prop === 'webdriver' && this === navigator) {
                    return false;
                }
                return originalHasOwnProperty.call(this, prop);
            };
            
            // Intercept property descriptor checks
            const originalGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
            Object.getOwnPropertyDescriptor = function(obj, prop) {
                if (prop === 'webdriver' && (obj === navigator || obj === Object.getPrototypeOf(navigator))) {
                    return undefined;
                }
                return originalGetOwnPropertyDescriptor.apply(this, arguments);
            };
            
            // Intercept 'in' operator checks by modifying the prototype chain
            const handler = {
                has(target, key) {
                    if (key === 'webdriver') return false;
                    return key in target;
                },
                get(target, key, receiver) {
                    if (key === 'webdriver') return undefined;
                    return Reflect.get(target, key, receiver);
                }
            };
        })();
        """
        
        # Inject webdriver evasion at CDP level - runs before page loads
        await self.cdp.send("Page.addScriptToEvaluateOnNewDocument", {
            "source": webdriver_evasion_script
        })
        
        # Also inject the main stealth script at CDP level for extra coverage
        await self.cdp.send("Page.addScriptToEvaluateOnNewDocument", {
            "source": stealth_script
        })
        
        # RESTORE: Mouse Tracking & Screencast
        self._mouse_x = 0
        self._mouse_y = 0
        self._mouse_down = False
        self.cdp.on("Page.screencastFrame", self._on_frame)

        await self.cdp.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": 60, # Reduced from 80 for speed
            "maxWidth": self.fingerprint["viewport"]["width"],
            "maxHeight": self.fingerprint["viewport"]["height"],
            "everyNthFrame": 2 # Reduced from 1 to halve CPU load while staying smooth
        })

        try:
            await self.page.goto("https://www.google.com", timeout=60000, wait_until="domcontentloaded")
        except Exception as e:
            logger.warning(f"Initial navigation timeout/error for {self.id}: {e}")
        
        self.is_active = True
        
        asyncio.create_task(self._frame_engine())
        logger.info(f"Browser {self.id} ({self.mode}) ready.")

    async def _frame_engine(self):
        """High-speed hybrid frame engine: CDP + Screenshot Fallback"""
        last_frame_count = 0
        while self.is_active:
            await asyncio.sleep(0.5) # Increased sleep from 0.1 to 0.5 to save CPU
            if self._frames_sent == last_frame_count:
                # CDP is stalled or not sending, force a screenshot
                try:
                    # Use lower quality for fallback screenshot to be faster
                    screenshot = await self.page.screenshot(type="jpeg", quality=40, scale="css")
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
node_status = "healthy"
node_error = None

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(heartbeat_loop())
    asyncio.create_task(command_loop())

async def command_loop():
    """Maintains a persistent websocket to the hub for receiving commands"""
    while True:
        try:
            ws_url = Config.SERVER_URL.replace("http", "ws") + f"/ws/node/{Config.NODE_ID}"
            logger.info(f"Connecting to command channel: {ws_url}")
            
            async with httpx.AsyncClient() as client:
                import websockets
                async with websockets.connect(
                    ws_url, 
                    ping_interval=20, 
                    ping_timeout=20,
                    close_timeout=10,
                    max_size=None 
                ) as ws:
                    logger.info("Command channel connected")
                    ws_lock = asyncio.Lock()
                    
                    async def handle_message(msg):
                        try:
                            data = json.loads(msg)
                            task_id = data.get("task_id")
                            command = data.get("command")
                            payload = data.get("data", {})
                            
                            logger.info(f"Processing Task [{command}]: {task_id}")
                            result = None
                            
                            try:
                                if command == "create":
                                    result = await create_browser_internal(
                                        payload.get("mode", "ephemeral"), 
                                        payload.get("profile_id"),
                                        payload.get("owner", "system")
                                    )
                                elif command == "get_profiles":
                                    result = await list_profiles()
                                elif command == "delete_profile":
                                    result = await delete_profile(payload.get("profile_id"))
                                elif command == "refresh":
                                    result = await node_refresh()
                                elif command == "close_browser":
                                    bid = payload.get("browser_id")
                                    if bid in browsers:
                                        await browsers[bid].cleanup()
                                        # Immediately remove from dict for accurate counting
                                        del browsers[bid]
                                        result = {"status": "ok", "browser_id": bid}
                                    else:
                                        result = {"status": "error", "message": "Browser not found"}
                            except Exception as te:
                                logger.error(f"Task Execution Error: {te}")
                                result = {"status": "error", "message": str(te)}
                            
                            async with ws_lock:
                                await ws.send(json.dumps({
                                    "task_id": task_id,
                                    "result": result
                                }))
                        except Exception as e:
                            logger.error(f"Message handling error: {e}")

                    while True:
                        msg = await ws.recv()
                        # Process each command in a detached task for concurrency
                        asyncio.create_task(handle_message(msg))
                        
        except Exception as e:
            logger.error(f"Command channel error: {e}")
            await asyncio.sleep(5)

async def create_browser_internal(mode: str = "ephemeral", profile_id: str = None, owner: str = "system"):
    """Helper for internal/websocket creation logic"""
    # Check limit before starting
    if len(browsers) >= Config.MAX_BROWSERS:
        return {"status": "error", "message": "Max browsers reached"}

    if mode == "persistent":
        active_profiles = {inst.profile_id for inst in browsers.values() if inst.mode == "persistent"}
        if profile_id and profile_id in active_profiles:
            return {"status": "error", "message": f"Profile {profile_id} is already in use"}
        
        if not profile_id:
            profiles_dir = os.path.abspath("profiles")
            if os.path.exists(profiles_dir):
                available_folders = [d for d in os.listdir(profiles_dir) if os.path.isdir(os.path.join(profiles_dir, d))]
                idle_profiles = [p for p in available_folders if p not in active_profiles]
                if idle_profiles: profile_id = idle_profiles[0]

    bid = str(uuid.uuid4())[:8]
    instance = BrowserInstance(bid, mode=mode, profile_id=profile_id, owner=owner)
    
    # Reserve slot immediately to prevent race conditions in concurrent launches
    browsers[bid] = instance
    
    try:
        await instance.launch()
        return {"id": bid, "mode": mode, "profile_id": instance.profile_id, "owner": owner}
    except Exception as e:
        # Rollback if launch fails
        if bid in browsers: del browsers[bid]
        raise e

# State tracking for delta updates
last_browser_ids = set()

async def heartbeat_loop():
    global node_status, node_error, last_browser_ids
    node_url = f"http://{Config.NODE_HOST}:{Config.NODE_PORT}"
    
    while True:
        try:
            current_browser_ids = set(browsers.keys())
            
            # Optimization: Only send full browser list if IDs changed
            should_send_list = (current_browser_ids != last_browser_ids)
            
            payload = {
                "node_id": Config.NODE_ID,
                "url": node_url,
                "browsers_count": len(browsers),
                "status": node_status,
                "error": node_error
            }
            
            if should_send_list:
                payload["browsers"] = [
                    {"id": b.id, "mode": b.mode, "profile_id": b.profile_id, "owner": b.owner}
                    for b in browsers.values()
                ]
                last_browser_ids = current_browser_ids

            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{Config.SERVER_URL}/register", json=payload)
                
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            node_status = "error"
            node_error = str(e)
            
        await asyncio.sleep(Config.HEARTBEAT_INTERVAL)

@app.post("/create")
async def create_browser(request: Request):
    try:
        try:
            data = await request.json()
        except:
            data = {}
            
        mode = data.get("mode", "ephemeral")
        profile_id = data.get("profile_id")
        
        if len(browsers) >= Config.MAX_BROWSERS:
            return {"status": "error", "message": "Max browsers reached"}

        # Logic for professional profile reuse
        if mode == "persistent":
            active_profiles = {inst.profile_id for inst in browsers.values() if inst.mode == "persistent"}
            
            if profile_id:
                if profile_id in active_profiles:
                     return {"status": "error", "message": f"Profile {profile_id} is already in use"}
            else:
                profiles_dir = os.path.abspath("profiles")
                if os.path.exists(profiles_dir):
                    available_folders = [d for d in os.listdir(profiles_dir) if os.path.isdir(os.path.join(profiles_dir, d))]
                    idle_profiles = [p for p in available_folders if p not in active_profiles]
                    if idle_profiles:
                        profile_id = idle_profiles[0]
                        logger.info(f"Reusing idle persistent profile: {profile_id}")

        bid = str(uuid.uuid4())[:8]
        instance = BrowserInstance(bid, mode=mode, profile_id=profile_id)
        await instance.launch()
        browsers[bid] = instance
        return {"id": bid, "mode": mode, "profile_id": instance.profile_id}
    except Exception as e:
        logger.error(f"API Create Error: {e}")
        global node_status, node_error
        node_status = "error"
        node_error = f"Spawn error: {str(e)}"
        return {"status": "error", "message": str(e)}

@app.post("/refresh")
async def node_refresh():
    """Manually clear node errors and force registration"""
    global node_status, node_error
    node_status = "healthy"
    node_error = None
    return {"status": "ok"}

@app.get("/profiles")
async def list_profiles():
    """List all stored profiles and their status"""
    profiles_dir = os.path.abspath("profiles")
    if not os.path.exists(profiles_dir):
        return {"profiles": []}
    
    active_profiles = {inst.profile_id: inst.id for inst in browsers.values() if inst.mode == "persistent"}
    
    results = []
    for d in os.listdir(profiles_dir):
        path = os.path.join(profiles_dir, d)
        if os.path.isdir(path):
            results.append({
                "profile_id": d,
                "is_active": d in active_profiles,
                "browser_id": active_profiles.get(d),
                "size": sum(f.stat().st_size for f in os.scandir(path) if f.is_file()) # Basic size calc
            })
    return {"profiles": results}

@app.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    """Delete a profile if it's not active"""
    active_profiles = {inst.profile_id for inst in browsers.values() if inst.mode == "persistent"}
    if profile_id in active_profiles:
        return {"status": "error", "message": "Cannot delete an active profile"}
    
    import shutil
    path = os.path.abspath(f"profiles/{profile_id}")
    if os.path.exists(path):
        shutil.rmtree(path)
        return {"status": "ok"}
    return {"status": "error", "message": "Profile not found"}

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
        sender.replaceTrack(track)
        
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
                if bid in browsers:
                    if pc in browsers[bid].pcs: browsers[bid].pcs.remove(pc)
                    if track in browsers[bid].tracks: browsers[bid].tracks.remove(track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    except Exception as e:
        logger.error(f"WebRTC Handshake Failed: {e}")
        # Safer Cleanup
        try:
            if bid in browsers:
                if pc in browsers[bid].pcs:
                    browsers[bid].pcs.remove(pc)
                if 'track' in locals() and track in browsers[bid].tracks:
                    browsers[bid].tracks.remove(track)
        except:
            pass
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
