#!/usr/bin/env python3
"""
=============================================================================
STEALTH NODE - ULTRA HIGH-PERFORMANCE BROWSER ORCHESTRATION ENGINE
=============================================================================
Designed for:
- 50,000+ concurrent browser instances
- 1,000+ simultaneous users
- Zero-downtime operation with auto-restart
- Memory-optimized with numpy acceleration
- Multi-threaded/multi-process architecture
- Circuit breaker pattern for resilience
- Self-healing capabilities
=============================================================================
"""

import os
import sys
import uuid
import time
import json
import asyncio
import base64
import platform
import subprocess
import signal
import gc
import traceback
import weakref
import atexit
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque
from functools import wraps
from contextlib import asynccontextmanager
from enum import Enum, auto
import numpy as np

# Third-party imports
import httpx
import cv2
import numpy as np
import threading
import logging
from logging.handlers import RotatingFileHandler

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer
from av import VideoFrame
import fractions
import webbrowser
import socket

# Optional Tray Icon imports
try:
    import pystray
    from pystray import MenuItem as item
    from PIL import Image, ImageDraw
except ImportError:
    pystray = None

# =============================================================================
# DATA DIRECTORY CONFIGURATION
# =============================================================================
# Prioritize local directory for portability, fallback to AppData if needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# =============================================================================
# LOGGING CONFIGURATION - Production Grade
# =============================================================================
def setup_logging():
    """Configure production-grade logging with rotation"""
    log_format = '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s'
    
    # Create logs directory
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler - handle case where sys.stdout might be None (compiled EXE)
    if sys.stdout is not None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(console_handler)
    
    # File handler with rotation (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        os.path.join(LOGS_DIR, "node.log"),
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Error file handler
    error_handler = RotatingFileHandler(
        os.path.join(LOGS_DIR, "node_errors.log"),
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(log_format))
    
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    return logging.getLogger("StealthNode")

logger = setup_logging()

# =============================================================================
# CONFIGURATION - Optimized for Scale
# =============================================================================
# Global state for command relay
active_tunnel_streams: Dict[str, str] = {} # browser_id -> stream_id
command_ws = None
command_ws_lock = asyncio.Lock()

@dataclass
class NodeConfig:
    """Centralized configuration with environment variable support"""
    NODE_ID: str = field(default_factory=lambda: os.getenv("NODE_ID", f"node-{str(uuid.uuid4())[:6]}"))
    SERVER_URL: str = field(default_factory=lambda: os.getenv("SERVER_URL", "https://server-latest-p0vo.onrender.com"))
    NODE_HOST: str = field(default_factory=lambda: os.getenv("NODE_HOST", "localhost"))
    NODE_PORT: int = field(default_factory=lambda: int(os.getenv("NODE_PORT", 8001)))
    HEADLESS: bool = field(default_factory=lambda: os.getenv("HEADLESS", "true").lower() == "true")
    
    # Scale settings - optimized for 50k browsers
    MAX_BROWSERS: int = field(default_factory=lambda: int(os.getenv("MAX_BROWSERS", 50000)))
    MAX_BROWSERS_PER_BATCH: int = field(default_factory=lambda: int(os.getenv("MAX_BROWSERS_PER_BATCH", 10)))
    
    # Thread/Process pools
    THREAD_POOL_SIZE: int = field(default_factory=lambda: int(os.getenv("THREAD_POOL_SIZE", min(32, (os.cpu_count() or 4) * 4))))
    PROCESS_POOL_SIZE: int = field(default_factory=lambda: int(os.getenv("PROCESS_POOL_SIZE", max(2, (os.cpu_count() or 4) // 2))))
    IO_THREAD_POOL_SIZE: int = field(default_factory=lambda: int(os.getenv("IO_THREAD_POOL_SIZE", min(64, (os.cpu_count() or 4) * 8))))
    
    # Timing
    HEARTBEAT_INTERVAL: int = 5
    HEALTH_CHECK_INTERVAL: int = 30
    CLEANUP_INTERVAL: int = 60
    RETRY_DELAY: float = 1.0
    MAX_RETRIES: int = 5
    
    # Video settings
    WIDTH: int = 1280
    HEIGHT: int = 720
    FPS: int = 30
    JPEG_QUALITY: int = 50
    
    # Memory management
    MEMORY_LIMIT_MB: int = field(default_factory=lambda: int(os.getenv("MEMORY_LIMIT_MB", 8192)))
    GC_THRESHOLD: int = field(default_factory=lambda: int(os.getenv("GC_THRESHOLD", 1000)))
    
    # Circuit breaker
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT: int = 30
    
    # Auto-restart
    AUTO_RESTART_ON_CRASH: bool = True
    MAX_RESTART_ATTEMPTS: int = 10
    RESTART_COOLDOWN: int = 5
    # Background task limits
    MAX_CONCURRENT_TASKS: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_TASKS", max(100, (os.cpu_count() or 4) * 25))))

    def get_browser_channel(self) -> Optional[str]:
        """Detect if Google Chrome is installed on the system"""
        if platform.system() == "Windows":
            paths = [
                os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
            ]
            for p in paths:
                if os.path.exists(p):
                    return "chrome"
        elif platform.system() == "Darwin": # macOS
            if os.path.exists("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
                return "chrome"
        elif platform.system() == "Linux":
            import shutil
            if shutil.which("google-chrome"):
                return "chrome"
    def get_local_ip(self) -> str:
        """Find the real network IP address of this machine"""
        # Try primary method: Connect to a dummy IP to see which interface is used
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # We don't actually send data, just "connect" to a public IP
                s.connect(("10.255.255.255", 1))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
        except Exception:
            pass
            
        # Fallback 1: Get by hostname
        try:
            ip = socket.gethostbyname(socket.gethostname())
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass

        # Fallback 2: List all interfaces (Windows/Linux/Mac)
        try:
            import socket
            hostname = socket.gethostname()
            # This can return multiple IPs, we try to find a non-local one
            ips = socket.gethostbyname_ex(hostname)[2]
            for ip in ips:
                if not ip.startswith("127."):
                    return ip
        except Exception:
            pass

        return "127.0.0.1"

Config = NodeConfig()

# Auto-detect IP if host is set to 'localhost' or default
if Config.NODE_HOST in ["localhost", "127.0.0.1"]:
    detected_ip = Config.get_local_ip()
    if detected_ip != "127.0.0.1":
        Config.NODE_HOST = detected_ip
        logger.info(f"NETWORK ACCESS ENABLED: Node is reachable at http://{Config.NODE_HOST}:{Config.NODE_PORT}")
    else:
        logger.warning("COULD NOT FIND LOCAL IP: Node is only reachable via 'localhost'. Remote PC access may fail.")

# Task limiter to bound concurrent background tasks created with create_limited_task
class TaskLimiter:
    def __init__(self, max_concurrent: int):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def create_limited_task(self, coro):
        async def _runner():
            await self._semaphore.acquire()
            try:
                await coro
            except Exception:
                # exceptions are logged by the coro itself
                pass
            finally:
                try:
                    self._semaphore.release()
                except Exception:
                    pass

        return asyncio.create_task(_runner())

task_limiter = TaskLimiter(Config.MAX_CONCURRENT_TASKS)

# =============================================================================
# NUMPY-OPTIMIZED IMAGE PROCESSING
# =============================================================================
class NumpyImageProcessor:
    """High-performance image processing using numpy vectorization"""
    
    # Pre-allocated buffers for common operations (reduces memory allocation overhead)
    _decode_buffer = None
    _encode_buffer = None
    _resize_buffer = None
    
    @staticmethod
    def initialize_buffers(width: int = 1280, height: int = 720):
        """Pre-allocate numpy buffers for zero-copy operations"""
        NumpyImageProcessor._decode_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        NumpyImageProcessor._encode_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        NumpyImageProcessor._resize_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        logger.info(f"Initialized numpy buffers: {width}x{height}")
    
    @staticmethod
    def fast_decode_jpeg(jpeg_bytes: bytes) -> Optional[np.ndarray]:
        """Decode JPEG with numpy optimization"""
        try:
            nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"JPEG decode error: {e}")
            return None
    
    @staticmethod
    def fast_encode_jpeg(img: np.ndarray, quality: int = 50) -> Optional[bytes]:
        """Encode to JPEG with numpy optimization"""
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, encoded = cv2.imencode('.jpg', img, encode_params)
            return encoded.tobytes()
        except Exception as e:
            logger.error(f"JPEG encode error: {e}")
            return None
    
    @staticmethod
    def fast_resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
        """Fast resize using numpy/cv2 optimization"""
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def bgr_to_rgb_inplace(img: np.ndarray) -> np.ndarray:
        """In-place BGR to RGB conversion using numpy views"""
        return img[:, :, ::-1]
    
    @staticmethod
    def batch_process_frames(frames: List[bytes], quality: int = 50) -> List[bytes]:
        """Process multiple frames in batch for efficiency"""
        results = []
        for frame_bytes in frames:
            img = NumpyImageProcessor.fast_decode_jpeg(frame_bytes)
            if img is not None:
                encoded = NumpyImageProcessor.fast_encode_jpeg(img, quality)
                if encoded:
                    results.append(encoded)
        return results

# Initialize buffers
NumpyImageProcessor.initialize_buffers(Config.WIDTH, Config.HEIGHT)

# =============================================================================
# THREAD POOL MANAGEMENT
# =============================================================================
class ThreadPoolManager:
    """Manages multiple thread pools for different workloads"""
    
    def __init__(self):
        self._pools: Dict[str, ThreadPoolExecutor] = {}
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._lock = threading.Lock()
        self._shutdown = False
        
    def get_pool(self, name: str, max_workers: int = None) -> ThreadPoolExecutor:
        """Get or create a named thread pool"""
        with self._lock:
            if name not in self._pools:
                workers = max_workers or Config.THREAD_POOL_SIZE
                self._pools[name] = ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix=f"pool_{name}_"
                )
                logger.info(f"Created thread pool '{name}' with {workers} workers")
            return self._pools[name]
    
    def get_process_pool(self) -> ProcessPoolExecutor:
        """Get or create the process pool for CPU-intensive tasks"""
        with self._lock:
            if self._process_pool is None:
                self._process_pool = ProcessPoolExecutor(
                    max_workers=Config.PROCESS_POOL_SIZE
                )
                logger.info(f"Created process pool with {Config.PROCESS_POOL_SIZE} workers")
            return self._process_pool
    
    def submit_io(self, fn: Callable, *args, **kwargs):
        """Submit I/O-bound task"""
        return self.get_pool("io", Config.IO_THREAD_POOL_SIZE).submit(fn, *args, **kwargs)
    
    def submit_cpu(self, fn: Callable, *args, **kwargs):
        """Submit CPU-bound task"""
        return self.get_pool("cpu", Config.THREAD_POOL_SIZE).submit(fn, *args, **kwargs)
    
    def submit_image(self, fn: Callable, *args, **kwargs):
        """Submit image processing task"""
        return self.get_pool("image", Config.THREAD_POOL_SIZE).submit(fn, *args, **kwargs)
    
    def shutdown(self, wait: bool = True):
        """Shutdown all pools"""
        with self._lock:
            self._shutdown = True
            for name, pool in self._pools.items():
                logger.info(f"Shutting down pool '{name}'")
                pool.shutdown(wait=wait)
            if self._process_pool:
                logger.info("Shutting down process pool")
                self._process_pool.shutdown(wait=wait)

# Global thread pool manager
pool_manager = ThreadPoolManager()

# =============================================================================
# CIRCUIT BREAKER PATTERN
# =============================================================================
class CircuitState(Enum):
    CLOSED = auto()  # Normal operation
    OPEN = auto()    # Failing, reject requests
    HALF_OPEN = auto()  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker for resilient service calls"""
    
    def __init__(self, name: str, threshold: int = 5, timeout: int = 30):
        self.name = name
        self.threshold = threshold
        self.timeout = timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
        self._lock = threading.Lock()
        
    def can_execute(self) -> bool:
        """Check if request can proceed"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit '{self.name}' entering HALF_OPEN state")
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def record_success(self):
        """Record successful execution"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 3:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit '{self.name}' CLOSED (recovered)")
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.success_count = 0
            
            if self.failure_count >= self.threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit '{self.name}' OPEN (threshold reached)")
    
    def __call__(self, func):
        """Decorator for circuit breaker"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not self.can_execute():
                raise Exception(f"Circuit '{self.name}' is OPEN")
            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise
        return wrapper

# Global circuit breakers
circuit_breakers = {
    "browser_launch": CircuitBreaker("browser_launch", threshold=5, timeout=30),
    "server_connection": CircuitBreaker("server_connection", threshold=3, timeout=60),
    "webrtc": CircuitBreaker("webrtc", threshold=5, timeout=30),
}

# =============================================================================
# RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# =============================================================================
def retry_async(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Async retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries} retries failed for {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator

def retry_sync(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """Sync retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries} retries failed for {func.__name__}: {e}")
            
            raise last_exception
        return wrapper
    return decorator

# =============================================================================
# MEMORY MANAGER
# =============================================================================
class MemoryManager:
    """Manages memory usage and triggers cleanup when needed"""
    
    def __init__(self, limit_mb: int = 8192):
        self.limit_bytes = limit_mb * 1024 * 1024
        self._last_gc = time.time()
        self._gc_count = 0
        self._lock = threading.Lock()
        
    def get_memory_usage(self) -> int:
        """Get current process memory usage in bytes"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # Fallback if psutil not available
            return 0
    
    def check_and_cleanup(self, force: bool = False) -> bool:
        """Check memory and trigger cleanup if needed"""
        with self._lock:
            current_usage = self.get_memory_usage()
            
            if force or current_usage > self.limit_bytes * 0.8:
                logger.warning(f"Memory cleanup triggered: {current_usage / 1024 / 1024:.1f}MB")
                
                # Force garbage collection
                gc.collect()
                self._gc_count += 1
                self._last_gc = time.time()
                
                # Clear numpy caches
                try:
                    np.finfo(np.float64).eps  # Trigger numpy cache clear
                except:
                    pass
                
                new_usage = self.get_memory_usage()
                freed = current_usage - new_usage
                logger.info(f"Memory freed: {freed / 1024 / 1024:.1f}MB, Current: {new_usage / 1024 / 1024:.1f}MB")
                
                return True
            return False
    
    def get_stats(self) -> dict:
        """Get memory statistics"""
        return {
            "current_mb": self.get_memory_usage() / 1024 / 1024,
            "limit_mb": self.limit_bytes / 1024 / 1024,
            "gc_count": self._gc_count,
            "last_gc": self._last_gc
        }

memory_manager = MemoryManager(Config.MEMORY_LIMIT_MB)

# =============================================================================
# HEALTH MONITOR
# =============================================================================
class HealthMonitor:
    """Monitors node health and triggers recovery actions"""
    
    def __init__(self):
        self.metrics = {
            "browsers_created": 0,
            "browsers_closed": 0,
            "errors": 0,
            "restarts": 0,
            "uptime_start": time.time(),
            "last_error": None,
            "last_error_time": None,
        }
        self._lock = threading.Lock()
        self._error_history = deque(maxlen=100)
        
    def record_browser_created(self):
        with self._lock:
            self.metrics["browsers_created"] += 1
    
    def record_browser_closed(self):
        with self._lock:
            self.metrics["browsers_closed"] += 1
    
    def record_error(self, error: str, context: str = ""):
        with self._lock:
            self.metrics["errors"] += 1
            self.metrics["last_error"] = error
            self.metrics["last_error_time"] = time.time()
            self._error_history.append({
                "time": time.time(),
                "error": error,
                "context": context
            })
    
    def record_restart(self):
        with self._lock:
            self.metrics["restarts"] += 1
    
    def get_health_status(self) -> dict:
        """Get comprehensive health status"""
        with self._lock:
            uptime = time.time() - self.metrics["uptime_start"]
            recent_errors = sum(1 for e in self._error_history if time.time() - e["time"] < 300)
            
            # Determine health level
            if recent_errors > 50:
                health = "critical"
            elif recent_errors > 20:
                health = "degraded"
            elif recent_errors > 5:
                health = "warning"
            else:
                health = "healthy"
            
            return {
                "status": health,
                "uptime_seconds": uptime,
                "browsers_active": self.metrics["browsers_created"] - self.metrics["browsers_closed"],
                "total_created": self.metrics["browsers_created"],
                "total_closed": self.metrics["browsers_closed"],
                "total_errors": self.metrics["errors"],
                "recent_errors_5min": recent_errors,
                "restarts": self.metrics["restarts"],
                "last_error": self.metrics["last_error"],
                "memory": memory_manager.get_stats(),
            }
    
    def should_restart(self) -> bool:
        """Determine if node should restart based on health"""
        with self._lock:
            recent_errors = sum(1 for e in self._error_history if time.time() - e["time"] < 60)
            return recent_errors > Config.CIRCUIT_BREAKER_THRESHOLD * 2

health_monitor = HealthMonitor()

# =============================================================================
# FINGERPRINT MANAGER
# =============================================================================
class FingerprintManager:
    """Generates unique browser fingerprints for evasion and multi-accounting"""
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    ]

    @staticmethod
    def generate() -> dict:
        """Generate a unique fingerprint using numpy for randomization"""
        # Use numpy for faster random generation
        rng = np.random.default_rng()
        
        cores = int(rng.choice([4, 8, 12, 16]))
        memory = int(rng.choice([4, 8, 16, 32]))
        
        width = 1280 + int(rng.integers(-20, 21))
        height = 720 + int(rng.integers(-20, 21))
        
        locales = ["en-US", "en-GB", "fr-FR", "de-DE", "es-ES"]
        timezones = ["America/New_York", "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Madrid"]
        idx = int(rng.integers(0, len(locales)))
        
        return {
            "user_agent": FingerprintManager.USER_AGENTS[int(rng.integers(0, len(FingerprintManager.USER_AGENTS)))],
            "viewport": {"width": width, "height": height},
            "device_memory": memory,
            "hardware_concurrency": cores,
            "locale": locales[idx],
            "timezone_id": timezones[idx],
            "webgl_vendor": ["Google Inc. (Intel)", "Google Inc. (NVIDIA)", "Google Inc. (AMD)"][int(rng.integers(0, 3))],
            "webgl_renderer": [
                "ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)",
                "ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0, D3D11)",
                "ANGLE (AMD, Radeon RX 6600 Direct3D11 vs_5_0 ps_5_0, D3D11)"
            ][int(rng.integers(0, 3))]
        }

# =============================================================================
# STEALTH JAVASCRIPT INJECTION
# =============================================================================
STEALTH_JS = r"""
(() => {
    // ============================================
    // 1. AGGRESSIVE WEBDRIVER EVASION
    // ============================================
    try {
        delete Object.getPrototypeOf(navigator).webdriver;
    } catch (e) {}
    
    Object.defineProperty(navigator, 'webdriver', {
        get: () => false,
        configurable: true,
        enumerable: false
    });
    
    const originalHasOwnProperty = Object.prototype.hasOwnProperty;
    Object.prototype.hasOwnProperty = function(prop) {
        if (prop === 'webdriver' && this === navigator) {
            return false;
        }
        return originalHasOwnProperty.call(this, prop);
    };
    
    const originalGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
    Object.getOwnPropertyDescriptor = function(obj, prop) {
        if (prop === 'webdriver' && obj === navigator) {
            return undefined;
        }
        return originalGetOwnPropertyDescriptor.apply(this, arguments);
    };
    
    const originalGetOwnPropertyNames = Object.getOwnPropertyNames;
    Object.getOwnPropertyNames = function(obj) {
        const props = originalGetOwnPropertyNames.apply(this, arguments);
        if (obj === navigator) {
            return props.filter(p => p !== 'webdriver');
        }
        return props;
    };
    
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
    // 4. Plugin Array Mocking
    // ============================================
    (function mockPlugins() {
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

        const pluginsArray = Object.create(PluginArray.prototype);
        const plugins = [pdfPlugin, chromePdfPlugin, chromiumPdfPlugin];
        
        plugins.forEach((plugin, i) => {
            Object.defineProperty(pluginsArray, i, { value: plugin, enumerable: true });
            Object.defineProperty(pluginsArray, plugin.name, { value: plugin, enumerable: false });
        });
        
        Object.defineProperty(pluginsArray, 'length', { value: plugins.length, enumerable: true });
        
        pluginsArray.item = function(index) { return plugins[index] || null; };
        pluginsArray.namedItem = function(name) { return plugins.find(p => p.name === name) || null; };
        pluginsArray.refresh = function() {};
        
        pluginsArray[Symbol.iterator] = function* () {
            for (let i = 0; i < plugins.length; i++) yield plugins[i];
        };

        Object.defineProperty(navigator, 'plugins', {
            get: () => pluginsArray,
            configurable: true
        });
        
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
    // 6. Chrome Object
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
    // 7. Smooth scrolling fix
    // ============================================
    window.addEventListener('wheel', (e) => {
        if (e.ctrlKey) e.preventDefault();
    }, { passive: false });
})();
"""

# =============================================================================
# BROWSER VIDEO TRACK
# =============================================================================
class BrowserVideoTrack(MediaStreamTrack):
    """WebRTC video track for browser streaming"""
    kind = "video"
    
    def __init__(self):
        super().__init__()
        self._queue = asyncio.Queue(maxsize=2)
        self._running = True

    async def recv(self):
        if not self._running:
            raise Exception("Track stopped")
        frame = await self._queue.get()
        return frame

    def push_frame(self, frame):
        if not self._running:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except:
                pass
        try:
            self._queue.put_nowait(frame)
        except:
            pass
    
    def stop(self):
        self._running = False
        super().stop()

# =============================================================================
# BROWSER INSTANCE - High Performance
# =============================================================================
class BrowserInstance:
    """High-performance browser instance with error handling and recovery"""
    
    __slots__ = [
        'id', 'mode', 'owner', 'profile_id', 'fingerprint', 'playwright',
        'browser', 'context', 'page', 'is_active', 'websockets', 'pcs',
        'tracks', '_last_pts', '_frames_sent', '_last_frame_log', 'input_lock',
        'profile_path', 'cdp', '_mouse_x', '_mouse_y', '_mouse_down',
        '_error_count', '_last_error', '_created_at', '_frame_buffer',
        '_cleanup_lock', '_is_cleaning', '_last_tunnel_frame'
    ]
    
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
        self.cdp = None
        self.is_active = False
        
        # Use weak references where possible to prevent memory leaks
        self.websockets: List[WebSocket] = []
        self.pcs: List[RTCPeerConnection] = []
        self.tracks: List[BrowserVideoTrack] = []
        
        self._last_pts = 0
        self._frames_sent = 0
        self._last_frame_log = time.time()
        self.input_lock = asyncio.Lock()
        
        # Error tracking
        self._error_count = 0
        self._last_error = None
        self._created_at = time.time()
        
        # Frame buffer for batch processing
        self._frame_buffer = deque(maxlen=5)
        
        # Cleanup state
        self._cleanup_lock = asyncio.Lock()
        self._is_cleaning = False
        
        # Path for persistent profiles - Use Non-privileged DATA_DIR
        self.profile_path = os.path.join(PROFILES_DIR, self.profile_id)
        if self.mode == "persistent":
            os.makedirs(self.profile_path, exist_ok=True)
            logger.info(f"Initialized persistent profile at {self.profile_path}")
        
        # Mouse state
        self._mouse_x = 0
        self._mouse_y = 0
        self._mouse_down = False
        
        # Tunnel stream rate limiting
        self._last_tunnel_frame = 0

    async def launch(self):
        """Launch browser with retry logic and error handling"""
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Clean up any previous failed attempt
                await self._cleanup_partial()
                
                logger.info(f"Launching {self.mode} browser {self.id} with profile {self.profile_id}... (attempt {attempt + 1}/{max_retries + 1})")
                await self._do_launch()
                return  # Success
                
            except Exception as e:
                last_error = e
                logger.error(f"Launch failed for {self.id}: {str(e)}")
                
                if attempt < max_retries:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for launch: {e}")
                    await asyncio.sleep(1.0 * (2 ** attempt))  # Exponential backoff
                else:
                    self._error_count += 1
                    self._last_error = str(e)
                    health_monitor.record_error(str(e), f"browser_launch_{self.id}")
                    raise last_error
    
    async def _cleanup_partial(self):
        """Clean up any partially initialized resources"""
        try:
            if self.cdp:
                try:
                    await self.cdp.detach()
                except:
                    pass
                self.cdp = None
            
            if self.page:
                try:
                    await self.page.close()
                except:
                    pass
                self.page = None
            
            if self.context:
                try:
                    await self.context.close()
                except:
                    pass
                self.context = None
            
            if self.browser:
                try:
                    await self.browser.close()
                except:
                    pass
                self.browser = None
            
            if self.playwright:
                try:
                    await self.playwright.stop()
                except:
                    pass
                self.playwright = None
        except:
            pass
    
    async def _do_launch(self):
        """Internal launch implementation"""
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
            "--enable-features=NetworkService,NetworkServiceInProcess",
            "--flag-switches-begin",
            "--flag-switches-end",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-breakpad",
            "--disable-component-update",
            "--disable-domain-reliability",
            "--disable-features=AudioServiceOutOfProcess",
            "--disable-ipc-flooding-protection",
            "--disable-renderer-backgrounding",
            "--force-color-profile=srgb",
            # Memory optimization flags
            "--js-flags=--max-old-space-size=512",
            "--disable-gpu-memory-buffer-video-frames",
            "--disable-gpu-vsync",
            "--disable-software-rasterizer",
            # Additional stability flags
            "--disable-crash-reporter",
            "--disable-in-process-stack-traces",
            "--disable-logging",
            "--log-level=3",
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

        if self.mode == "persistent":
            self.profile_path = os.path.normpath(self.profile_path)
            os.makedirs(self.profile_path, exist_ok=True)
            
            self.context = await self.playwright.chromium.launch_persistent_context(
                user_data_dir=self.profile_path,
                headless=Config.HEADLESS,
                args=launch_args,
                channel=Config.get_browser_channel(), # Use system chrome if available
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
            self.browser = await self.playwright.chromium.launch(
                headless=Config.HEADLESS,
                args=launch_args,
                channel=Config.get_browser_channel() # Use system chrome if available
            )
            self.context = await self.browser.new_context(**context_params)
            self.page = await self.context.new_page()
        
        # Dynamic Stealth Injection with Fingerprint
        stealth_script = STEALTH_JS.replace("'Google Inc. (Intel)'", f"'{self.fingerprint['webgl_vendor']}'")
        stealth_script = stealth_script.replace(
            "'ANGLE (Intel, Intel(R) UHD Graphics Direct3D11 vs_5_0 ps_5_0, D3D11)'",
            f"'{self.fingerprint['webgl_renderer']}'"
        )
        stealth_script = stealth_script.replace(
            "deviceMemory', { get: () => 8 }",
            f"deviceMemory', {{ get: () => {self.fingerprint['device_memory']} }}"
        )
        stealth_script = stealth_script.replace(
            "hardwareConcurrency', { get: () => 8 }",
            f"hardwareConcurrency', {{ get: () => {self.fingerprint['hardware_concurrency']} }}"
        )

        await self.context.add_init_script(stealth_script)
        
        self.cdp = await self.context.new_cdp_session(self.page)
        logger.info(f"CDP Session started for {self.id}")
        
        # CDP-level webdriver evasion
        webdriver_evasion_script = """
        (() => {
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
                configurable: true
            });
            
            try {
                const proto = Object.getPrototypeOf(navigator);
                if (proto.hasOwnProperty('webdriver')) {
                    delete proto.webdriver;
                }
            } catch(e) {}
            
            const originalHasOwnProperty = Object.prototype.hasOwnProperty;
            Object.prototype.hasOwnProperty = function(prop) {
                if (prop === 'webdriver' && this === navigator) {
                    return false;
                }
                return originalHasOwnProperty.call(this, prop);
            };
            
            const originalGetOwnPropertyDescriptor = Object.getOwnPropertyDescriptor;
            Object.getOwnPropertyDescriptor = function(obj, prop) {
                if (prop === 'webdriver' && (obj === navigator || obj === Object.getPrototypeOf(navigator))) {
                    return undefined;
                }
                return originalGetOwnPropertyDescriptor.apply(this, arguments);
            };
        })();
        """
        
        await self.cdp.send("Page.addScriptToEvaluateOnNewDocument", {
            "source": webdriver_evasion_script
        })
        
        await self.cdp.send("Page.addScriptToEvaluateOnNewDocument", {
            "source": stealth_script
        })
        
        # Setup screencast - use lambda wrapper to properly handle async callback
        def on_frame_handler(data):
            task_limiter.create_limited_task(self._on_frame(data))
        
        self.cdp.on("Page.screencastFrame", on_frame_handler)

        await self.cdp.send("Page.startScreencast", {
            "format": "jpeg",
            "quality": 60,
            "maxWidth": self.fingerprint["viewport"]["width"],
            "maxHeight": self.fingerprint["viewport"]["height"],
            "everyNthFrame": 2
        })

        try:
            await self.page.goto("https://www.google.com", timeout=60000, wait_until="domcontentloaded")
        except Exception as e:
            logger.warning(f"Initial navigation timeout/error for {self.id}: {e}")
        
        self.is_active = True
        
        asyncio.create_task(self._frame_engine())
        asyncio.create_task(self._health_check_loop())
        
        health_monitor.record_browser_created()
        logger.info(f"Browser {self.id} ({self.mode}) ready.")

    async def _health_check_loop(self):
        """Periodic health check for the browser instance"""
        while self.is_active:
            try:
                await asyncio.sleep(Config.HEALTH_CHECK_INTERVAL)
                
                if not self.is_active:
                    break
                
                # Check if page is still responsive
                if self.page:
                    try:
                        await asyncio.wait_for(
                            self.page.evaluate("() => true"),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Browser {self.id} health check timeout")
                        self._error_count += 1
                    except Exception as e:
                        logger.warning(f"Browser {self.id} health check failed: {e}")
                        self._error_count += 1
                
                # Check error threshold
                if self._error_count > 10:
                    logger.error(f"Browser {self.id} exceeded error threshold, marking for cleanup")
                    self.is_active = False
                    
            except Exception as e:
                logger.error(f"Health check loop error for {self.id}: {e}")

    async def _frame_engine(self):
        """High-speed hybrid frame engine with error recovery"""
        last_frame_count = 0
        consecutive_failures = 0
        
        while self.is_active:
            try:
                await asyncio.sleep(0.5)
                
                if not self.is_active:
                    break
                
                if self._frames_sent == last_frame_count:
                    # CDP is stalled, force a screenshot
                    try:
                        screenshot = await asyncio.wait_for(
                            self.page.screenshot(type="jpeg", quality=40, scale="css"),
                            timeout=5.0
                        )
                        # screenshot is raw bytes; pass bytes to distributor to avoid extra base64 churn
                        await self._distribute_frame(screenshot)
                        consecutive_failures = 0
                    except asyncio.TimeoutError:
                        consecutive_failures += 1
                        logger.warning(f"Screenshot timeout for {self.id}")
                    except Exception as e:
                        consecutive_failures += 1
                        logger.error(f"Screenshot error for {self.id}: {e}")
                
                last_frame_count = self._frames_sent
                
                # Check for too many consecutive failures
                if consecutive_failures > 5:
                    logger.error(f"Too many frame failures for {self.id}")
                    self._error_count += 5
                    consecutive_failures = 0
                    
            except Exception as e:
                logger.error(f"Frame engine error for {self.id}: {e}")
                await asyncio.sleep(1)

    async def _distribute_frame(self, b64_data: str):
        """Distribute frame to all connected clients"""
        self._frames_sent += 1
        now = time.time()
        
        if now - self._last_frame_log > 5:
            logger.debug(f"NODE FLOW [{self.id}]: {self._frames_sent} frames delivered in last 5s")
            self._frames_sent = 0
            self._last_frame_log = now

        # Distribute to WebSocket clients
        if self.websockets:
            # If we received raw bytes, encode to base64 only when sending to clients
            if isinstance(b64_data, (bytes, bytearray)):
                send_b64 = base64.b64encode(b64_data).decode('utf-8')
            else:
                send_b64 = b64_data

            msg = json.dumps({"type": "frame", "data": send_b64})
            dead_sockets = []

            for ws in list(self.websockets):
                try:
                    await asyncio.wait_for(ws.send_text(msg), timeout=1.0)
                except Exception:
                    dead_sockets.append(ws)

            # Clean up dead sockets
            for ws in dead_sockets:
                if ws in self.websockets:
                    self.websockets.remove(ws)

        # TUNNEL RELAY: If this browser has an active stream to the hub
        # Rate limit to prevent server overload (max 10 fps through relay)
        global command_ws, active_tunnel_streams
        if self.id in active_tunnel_streams and command_ws:
            now = time.time()
            if now - self._last_tunnel_frame >= 0.1:  # 100ms = 10fps
                self._last_tunnel_frame = now
                stream_id = active_tunnel_streams[self.id]
                
                if isinstance(b64_data, (bytes, bytearray)):
                    send_b64 = base64.b64encode(b64_data).decode('utf-8')
                else:
                    send_b64 = b64_data
                    
                msg = {
                    "task_id": stream_id, # Hub uses task_id starting with stream_ to route
                    "result": {"type": "frame", "data": send_b64}
                }
                
                asyncio.create_task(self._send_to_command_ws(msg))

        # Distribute to WebRTC tracks — hand raw bytes to image worker when possible
        if self.tracks:
            if isinstance(b64_data, (bytes, bytearray)):
                pool_manager.submit_image(self._process_and_push_webrtc, b64_data)
            else:
                # still accept base64 strings for backward compatibility
                pool_manager.submit_image(self._process_and_push_webrtc, b64_data)

    async def _on_frame(self, data):
        """Handle CDP screencast frame"""
        if not self.is_active:
            return
        try:
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": data["sessionId"]})
            # Decode base64 once and pass raw bytes internally to avoid repeated decode/encode
            try:
                raw = base64.b64decode(data["data"])
            except Exception:
                raw = data["data"]
            await self._distribute_frame(raw)
        except Exception as e:
            logger.error(f"Frame handling error for {self.id}: {e}")

    async def _send_to_command_ws(self, msg: dict):
        """Helper to send data via the command WebSocket without blocking"""
        global command_ws, command_ws_lock
        if not command_ws:
            return
        async with command_ws_lock:
            try:
                await command_ws.send(json.dumps(msg))
            except Exception:
                pass

    def _process_and_push_webrtc(self, raw_bytes):
        """Process frame for WebRTC (runs in thread pool). Accepts raw bytes or base64 string."""
        try:
            if isinstance(raw_bytes, (bytes, bytearray)):
                rb = raw_bytes
            else:
                try:
                    rb = base64.b64decode(raw_bytes)
                except Exception:
                    return

            img = NumpyImageProcessor.fast_decode_jpeg(rb)

            if img is None:
                return

            img_rgb = NumpyImageProcessor.bgr_to_rgb_inplace(img)
            frame = VideoFrame.from_ndarray(img_rgb, format='rgb24')
            frame.pts = self._last_pts
            frame.time_base = fractions.Fraction(1, 90000)
            self._last_pts += 3000

            for track in list(self.tracks):
                try:
                    track.push_frame(frame)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"WebRTC frame processing error: {e}")

    def _bezier_curve(self, start_x, start_y, end_x, end_y, steps=20):
        """Generate smooth Bézier curve points using numpy"""
        # Use numpy for vectorized computation
        t = np.linspace(0, 1, steps + 1)
        
        # Random control points
        rng = np.random.default_rng()
        ctrl1_x = start_x + rng.uniform(-50, 50) + (end_x - start_x) * 0.3
        ctrl1_y = start_y + rng.uniform(-50, 50) + (end_y - start_y) * 0.3
        ctrl2_x = start_x + rng.uniform(-50, 50) + (end_x - start_x) * 0.7
        ctrl2_y = start_y + rng.uniform(-50, 50) + (end_y - start_y) * 0.7
        
        # Vectorized Bézier calculation
        x = (1-t)**3 * start_x + 3*(1-t)**2*t * ctrl1_x + 3*(1-t)*t**2 * ctrl2_x + t**3 * end_x
        y = (1-t)**3 * start_y + 3*(1-t)**2*t * ctrl1_y + 3*(1-t)*t**2 * ctrl2_y + t**3 * end_y
        
        # Add jitter
        jitter_x = rng.uniform(-1.5, 1.5, len(t))
        jitter_y = rng.uniform(-1.5, 1.5, len(t))
        
        x = np.clip(x + jitter_x, 0, Config.WIDTH).astype(int)
        y = np.clip(y + jitter_y, 0, Config.HEIGHT).astype(int)
        
        return list(zip(x, y))

    async def _human_move(self, from_x, from_y, to_x, to_y, is_dragging=False):
        """Move mouse in a human-like way"""
        distance = np.sqrt((to_x - from_x)**2 + (to_y - from_y)**2)
        
        if distance < 10:
            await self.page.mouse.move(to_x, to_y)
            return
        
        if is_dragging:
            steps = max(15, min(35, int(distance / 10)))
        else:
            steps = max(5, min(20, int(distance / 20)))
        
        path = self._bezier_curve(from_x, from_y, to_x, to_y, steps)
        rng = np.random.default_rng()
        
        for i, (px, py) in enumerate(path):
            progress = i / len(path)
            
            if is_dragging:
                if progress < 0.1:
                    delay = rng.uniform(0.008, 0.015)
                elif progress > 0.85:
                    delay = rng.uniform(0.010, 0.020)
                else:
                    delay = rng.uniform(0.003, 0.008)
            else:
                delay = rng.uniform(0.002, 0.006)
            
            await self.page.mouse.move(int(px), int(py))
            await asyncio.sleep(delay)
        
        if is_dragging and distance > 50:
            overshoot_x = to_x + rng.integers(-2, 3)
            overshoot_y = to_y + rng.integers(-2, 3)
            await self.page.mouse.move(int(overshoot_x), int(overshoot_y))
            await asyncio.sleep(rng.uniform(0.015, 0.030))
            await self.page.mouse.move(to_x, to_y)
            await asyncio.sleep(rng.uniform(0.005, 0.015))

    async def handle_input(self, action: dict):
        """Handle user input with error recovery"""
        if not self.is_active:
            return
        
        async with self.input_lock:
            try:
                atype = action.get("type")
                x, y = action.get("x", 0), action.get("y", 0)
                
                if x is not None:
                    x = max(0, min(int(x), Config.WIDTH))
                if y is not None:
                    y = max(0, min(int(y), Config.HEIGHT))
                
                button = action.get("button", "left")
                
                prev_x, prev_y = self._mouse_x, self._mouse_y
                self._mouse_x, self._mouse_y = x, y
                
                rng = np.random.default_rng()

                if atype == "mousemove":
                    if self._mouse_down:
                        await self.page.mouse.move(x, y, steps=1)
                    else:
                        await self.page.mouse.move(x, y, steps=1)
                
                elif atype == "mousedown":
                    if abs(x - prev_x) > 5 or abs(y - prev_y) > 5:
                        await self._human_move(prev_x, prev_y, x, y, is_dragging=False)
                    
                    await asyncio.sleep(rng.uniform(0.020, 0.050))
                    self._mouse_down = True
                    await self.page.mouse.down(button=button)
                
                elif atype == "mouseup":
                    await asyncio.sleep(rng.uniform(0.015, 0.040))
                    self._mouse_down = False
                    await self.page.mouse.up(button=button)

                elif atype == "click":
                    if abs(x - prev_x) > 5 or abs(y - prev_y) > 5:
                        await self._human_move(prev_x, prev_y, x, y, is_dragging=False)
                    
                    await asyncio.sleep(rng.uniform(0.030, 0.070))
                    await self.page.mouse.down(button=button)
                    await asyncio.sleep(rng.uniform(0.050, 0.120))
                    await self.page.mouse.up(button=button)
                    await asyncio.sleep(rng.uniform(0.010, 0.030))
                
                elif atype == "key":
                    key = action.get("key")
                    if key:
                        await self.page.keyboard.down(key)
                        await asyncio.sleep(rng.uniform(0.030, 0.080))
                        await self.page.keyboard.up(key)
                
                elif atype == "scroll":
                    await self.page.mouse.wheel(0, int(action.get("deltaY", 0)))
                
                elif atype == "navigate":
                    url = action["url"]
                    if not url.startswith("http"):
                        url = "https://" + url
                    await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                elif atype == "refresh":
                    await self.page.reload()
                elif atype == "back":
                    await self.page.go_back()
                elif atype == "forward":
                    await self.page.go_forward()
                    
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                logger.error(f"Input Error {self.id}: {e}")

    async def cleanup(self):
        """Clean up browser resources with proper error handling"""
        async with self._cleanup_lock:
            if self._is_cleaning:
                return
            self._is_cleaning = True
        
        self.is_active = False
        logger.info(f"Cleaning up browser {self.id}")
        
        try:
            # Stop all tracks
            for track in list(self.tracks):
                try:
                    track.stop()
                except:
                    pass
            self.tracks.clear()
            
            # Close all peer connections
            for pc in list(self.pcs):
                try:
                    await pc.close()
                except:
                    pass
            self.pcs.clear()
            
            # Close websockets
            for ws in list(self.websockets):
                try:
                    await ws.close()
                except:
                    pass
            self.websockets.clear()
            
            # Close browser components
            if self.cdp:
                try:
                    await self.cdp.detach()
                except:
                    pass
            
            if self.page:
                try:
                    await self.page.close()
                except:
                    pass
            
            if self.context:
                try:
                    await self.context.close()
                except:
                    pass
            
            if self.browser:
                try:
                    await self.browser.close()
                except:
                    pass
            
            if self.playwright:
                try:
                    await self.playwright.stop()
                except:
                    pass
            
            health_monitor.record_browser_closed()
            logger.info(f"Browser {self.id} cleanup complete")
            
        except Exception as e:
            logger.error(f"Cleanup error for {self.id}: {e}")

# =============================================================================
# BROWSER MANAGER - Handles 50k+ browsers
# =============================================================================
class BrowserManager:
    """High-performance browser manager for massive scale"""
    
    def __init__(self):
        self._browsers: Dict[str, BrowserInstance] = {}
        self._lock = asyncio.Lock()
        self._creation_semaphore = asyncio.Semaphore(Config.MAX_BROWSERS_PER_BATCH)
        self._cleanup_task = None
        
    @property
    def browsers(self) -> Dict[str, BrowserInstance]:
        return self._browsers
    
    def __len__(self) -> int:
        return len(self._browsers)
    
    def __contains__(self, bid: str) -> bool:
        return bid in self._browsers
    
    def __getitem__(self, bid: str) -> BrowserInstance:
        return self._browsers[bid]
    
    def get(self, bid: str) -> Optional[BrowserInstance]:
        return self._browsers.get(bid)
    
    def values(self):
        return self._browsers.values()
    
    def keys(self):
        return self._browsers.keys()
    
    def items(self):
        return self._browsers.items()
    
    async def create(self, mode: str = "ephemeral", profile_id: str = None, owner: str = "system") -> dict:
        """Create a new browser instance with rate limiting"""
        async with self._creation_semaphore:
            if len(self._browsers) >= Config.MAX_BROWSERS:
                return {"status": "error", "message": f"Max browsers ({Config.MAX_BROWSERS}) reached"}
            
            if mode == "persistent":
                active_profiles = {inst.profile_id for inst in self._browsers.values() if inst.mode == "persistent"}
                if profile_id and profile_id in active_profiles:
                    return {"status": "error", "message": f"Profile {profile_id} is already in use"}
                
                if not profile_id:
                    if os.path.exists(PROFILES_DIR):
                        available_folders = [d for d in os.listdir(PROFILES_DIR) if os.path.isdir(os.path.join(PROFILES_DIR, d))]
                        idle_profiles = [p for p in available_folders if p not in active_profiles]
                        if idle_profiles:
                            profile_id = idle_profiles[0]
            
            bid = str(uuid.uuid4())[:8]
            instance = BrowserInstance(bid, mode=mode, profile_id=profile_id, owner=owner)
            
            # Reserve slot immediately
            async with self._lock:
                self._browsers[bid] = instance
            
            try:
                # Add a small staggered delay based on active creations to prevent thundering herd
                # but only if we are under heavy load
                active_creations = Config.MAX_BROWSERS_PER_BATCH - self._creation_semaphore._value
                if active_creations > 2:
                    await asyncio.sleep(0.25 * active_creations)

                await instance.launch()
                return {"id": bid, "mode": mode, "profile_id": instance.profile_id, "owner": owner}
            except Exception as e:
                async with self._lock:
                    if bid in self._browsers:
                        del self._browsers[bid]
                raise e
    
    async def close(self, bid: str) -> dict:
        """Close a browser instance"""
        if bid not in self._browsers:
            return {"status": "error", "message": "Browser not found"}
        
        instance = self._browsers[bid]
        await instance.cleanup()
        
        async with self._lock:
            if bid in self._browsers:
                del self._browsers[bid]
        
        return {"status": "ok", "browser_id": bid}
    
    async def close_all(self):
        """Close all browser instances"""
        tasks = []
        for bid in list(self._browsers.keys()):
            tasks.append(self.close(bid))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def cleanup_dead_browsers(self):
        """Clean up browsers that are no longer active"""
        dead_browsers = []
        
        for bid, instance in list(self._browsers.items()):
            if not instance.is_active or instance._error_count > 10:
                dead_browsers.append(bid)
        
        for bid in dead_browsers:
            logger.info(f"Cleaning up dead browser: {bid}")
            await self.close(bid)
        
        return len(dead_browsers)
    
    def start_cleanup_loop(self):
        """Start the periodic cleanup loop"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(Config.CLEANUP_INTERVAL)
                    cleaned = await self.cleanup_dead_browsers()
                    if cleaned > 0:
                        logger.info(f"Cleaned up {cleaned} dead browsers")
                    
                    # Trigger memory cleanup
                    memory_manager.check_and_cleanup()
                    
                except Exception as e:
                    logger.error(f"Cleanup loop error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())

# Global browser manager
browser_manager = BrowserManager()

# =============================================================================
# WEBRTC SIGNALING HELPERS FOR P2P THROUGH SERVER
# =============================================================================
# When hosting on Render/HuggingFace, direct WebRTC connections can't be initiated
# because the UI can't reach the node directly. Instead, signaling goes through
# the server's command channel, but actual media/data flows P2P once connected.

# Track pending ICE candidates per browser (for trickle ICE)
pending_ice_candidates: Dict[str, List[dict]] = {}

async def handle_webrtc_offer_internal(bid: str, offer_data: dict) -> dict:
    """
    Handle a WebRTC offer for a browser, coming through the server command channel.
    Returns the answer to be sent back through the server.
    """
    try:
        if bid not in browser_manager:
            raise ValueError(f"Browser {bid} not found")
        
        logger.info(f"[P2P Signaling] Processing WebRTC offer for {bid}")
        
        # Create RTCPeerConnection with multiple STUN servers for better NAT traversal
        pc = RTCPeerConnection(configuration=RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun2.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun3.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun4.l.google.com:19302"]),
            ]
        ))
        browser_manager[bid].pcs.append(pc)
        
        # Create video track
        track = BrowserVideoTrack()
        browser_manager[bid].tracks.append(track)
        
        pc.addTransceiver("video", direction="sendonly")
        sender = pc.getSenders()[0]
        sender.replaceTrack(track)
        
        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"[P2P] DataChannel created for {bid}")
            
            @channel.on("message")
            def on_message(message):
                try:
                    action = json.loads(message)
                    task_limiter.create_limited_task(browser_manager[bid].handle_input(action))
                except Exception as e:
                    logger.error(f"DC Parse Error: {e}")
        
        @pc.on("connectionstatechange")
        async def on_state():
            logger.info(f"[P2P] Connection state for {bid}: {pc.connectionState}")
            if pc.connectionState == "connected":
                logger.info(f"[P2P] ✓ P2P connection established for {bid}")
            elif pc.connectionState in ["failed", "closed", "disconnected"]:
                if bid in browser_manager:
                    if pc in browser_manager[bid].pcs:
                        browser_manager[bid].pcs.remove(pc)
                    if track in browser_manager[bid].tracks:
                        browser_manager[bid].tracks.remove(track)
        
        # Set remote description from offer
        offer = RTCSessionDescription(sdp=offer_data["sdp"], type=offer_data["type"])
        await pc.setRemoteDescription(offer)
        
        # Apply any pending ICE candidates
        if bid in pending_ice_candidates:
            logger.info(f"[P2P Signaling] Applying {len(pending_ice_candidates[bid])} pending ICE candidates for {bid}")
            for candidate_data in pending_ice_candidates[bid]:
                try:
                    if candidate_data.get("candidate"):
                        await pc.addIceCandidate(candidate_data)
                except Exception as e:
                    logger.warning(f"Failed to add pending ICE candidate: {e}")
            del pending_ice_candidates[bid]
        
        # Create and set local description
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        # Wait for ICE gathering to complete for better connectivity in restricted networks
        try:
            wait_start = time.time()
            while pc.iceGatheringState != 'complete' and time.time() - wait_start < 3.0:
                await asyncio.sleep(0.1)
            logger.info(f"[P2P Signaling] ICE gathering for {bid} finished in {time.time() - wait_start:.2f}s with state: {pc.iceGatheringState}")
        except Exception as e:
            logger.warning(f"[P2P Signaling] Error during ICE gathering wait: {e}")
        
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    except Exception as e:
        logger.error(f"[P2P Signaling] CRITICAL ERROR in handle_webrtc_offer_internal: {e}")
        logger.error(traceback.format_exc())
        raise e

async def handle_ice_candidate_internal(bid: str, candidate_data: dict):
    """
    Handle an ICE candidate for a browser, coming through the server command channel.
    """
    if bid not in browser_manager:
        return
    
    # Find the peer connection for this browser
    browser = browser_manager[bid]
    if browser.pcs:
        pc = browser.pcs[-1]  # Use the most recent peer connection
        try:
            if candidate_data.get("candidate"):
                await pc.addIceCandidate(candidate_data)
                logger.debug(f"[P2P] Added ICE candidate for {bid}")
        except Exception as e:
            logger.warning(f"Failed to add ICE candidate for {bid}: {e}")
    else:
        # Store for later if no PC yet
        if bid not in pending_ice_candidates:
            pending_ice_candidates[bid] = []
        pending_ice_candidates[bid].append(candidate_data)

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Stealth Node...")
    supervisor.loop = asyncio.get_running_loop()
    
    # Start background tasks
    asyncio.create_task(heartbeat_loop())
    asyncio.create_task(command_loop())
    browser_manager.start_cleanup_loop()
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Stealth Node...")
    await browser_manager.close_all()
    pool_manager.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# State tracking
node_status = "healthy"
node_error = None
last_browser_ids: Set[str] = set()

# =============================================================================
# PROFESSIONAL CONNECTION MANAGEMENT FOR NODE
# =============================================================================

class ConnectionState(Enum):
    """Represents the state of the node's connection to the hub"""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()
    BACKOFF = auto()

@dataclass
class HubConnection:
    """Tracks the connection state to the hub"""
    state: ConnectionState = ConnectionState.DISCONNECTED
    websocket: Optional[any] = None
    last_connected: float = 0
    last_attempt: float = 0
    reconnect_attempts: int = 0
    consecutive_failures: int = 0
    backoff_delay: float = 1.0
    max_backoff_delay: float = 300.0  # 5 minutes
    base_delay: float = 1.0
    max_attempts: int = 100
    last_error: Optional[str] = None

    def should_attempt_reconnect(self) -> bool:
        """Determine if we should attempt to reconnect"""
        now = time.time()

        # Don't attempt if we're already connected
        if self.state == ConnectionState.CONNECTED:
            return False

        # Check if we've exceeded max attempts
        if self.reconnect_attempts >= self.max_attempts:
            logger.error(f"Exceeded maximum reconnection attempts ({self.max_attempts})")
            return False

        # Check backoff delay
        if now - self.last_attempt < self.backoff_delay:
            return False

        return True

    def record_success(self):
        """Record a successful connection"""
        self.state = ConnectionState.CONNECTED
        self.last_connected = time.time()
        self.consecutive_failures = 0
        self.reconnect_attempts = 0
        self.backoff_delay = self.base_delay
        self.last_error = None
        logger.info("Hub connection established successfully")

    def record_failure(self, error: str):
        """Record a connection failure"""
        self.state = ConnectionState.RECONNECTING
        self.consecutive_failures += 1
        self.reconnect_attempts += 1
        self.last_attempt = time.time()
        self.last_error = error

        # Exponential backoff with jitter
        self.backoff_delay = min(
            self.base_delay * (2 ** min(self.consecutive_failures - 1, 10)),
            self.max_backoff_delay
        )

        # Add jitter (±25%)
        jitter = self.backoff_delay * 0.25 * (np.random.random() - 0.5) * 2
        self.backoff_delay = max(0.1, self.backoff_delay + jitter)

        logger.warning(f"Hub connection failed (attempt {self.reconnect_attempts}): {error}")
        logger.info(f"Next reconnection attempt in {self.backoff_delay:.1f}s")

    def get_status(self) -> Dict[str, Any]:
        """Get connection status for monitoring"""
        return {
            "state": self.state.name,
            "last_connected": self.last_connected,
            "last_attempt": self.last_attempt,
            "reconnect_attempts": self.reconnect_attempts,
            "consecutive_failures": self.consecutive_failures,
            "backoff_delay": self.backoff_delay,
            "last_error": self.last_error,
            "time_since_last_attempt": time.time() - self.last_attempt,
            "time_since_last_success": time.time() - self.last_connected if self.last_connected > 0 else None
        }

# Global hub connection tracker
hub_connection = HubConnection()

# =============================================================================
# BACKGROUND TASKS
# =============================================================================
async def command_loop():
    """Maintains a persistent websocket to the hub for receiving commands"""
    global node_status, node_error, command_ws
    reconnect_delay = hub_connection.base_delay
    max_reconnect_delay = hub_connection.max_backoff_delay

    while True:
        try:
            ws_url = Config.SERVER_URL.replace("http", "ws") + f"/ws/node/{Config.NODE_ID}"
            logger.info(f"Connecting to command channel: {ws_url}")

            import websockets
            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                max_size=None
            ) as ws:
                command_ws = ws
                logger.info("Command channel connected")
                # Mark success and reset backoff
                hub_connection.record_success()
                node_status = "healthy"
                node_error = None
                reconnect_delay = hub_connection.backoff_delay

                # Immediately register with the hub to ensure server mapping is up-to-date
                try:
                    payload = {
                        "node_id": Config.NODE_ID,
                        "url": f"http://{Config.NODE_HOST}:{Config.NODE_PORT}",
                        "browsers_count": len(browser_manager),
                        "status": node_status,
                        "error": node_error,
                        "health": health_monitor.get_health_status()
                    }
                    payload["browsers"] = [
                        {"id": b.id, "mode": b.mode, "profile_id": b.profile_id, "owner": b.owner}
                        for b in browser_manager.values()
                    ]
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        await client.post(f"{Config.SERVER_URL}/register", json=payload)
                except Exception as e:
                    logger.debug(f"Initial register failed: {e}")

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
                                result = await browser_manager.create(
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
                                result = await browser_manager.close(bid)
                            elif command == "health":
                                result = health_monitor.get_health_status()
                            elif command == "restart":
                                asyncio.create_task(supervisor.shutdown_and_restart())
                                result = {"status": "ok", "message": "Restarting..."}
                            elif command == "start_stream":
                                bid = payload.get("browser_id")
                                stream_id = payload.get("stream_id")
                                active_tunnel_streams[bid] = stream_id
                                logger.info(f"Started tunnel stream for {bid} -> {stream_id}")
                                result = {"status": "ok"}
                            elif command == "stop_stream":
                                bid = payload.get("browser_id")
                                if bid in active_tunnel_streams:
                                    del active_tunnel_streams[bid]
                                result = {"status": "ok"}
                            elif command == "browser_action":
                                bid = payload.get("browser_id")
                                action = payload.get("action")
                                if bid in browser_manager:
                                    await browser_manager[bid].handle_input(action)
                                result = {"status": "ok"}
                            elif command == "rtc_offer":
                                # WebRTC signaling: handle offer from UI through server
                                bid = payload.get("browser_id")
                                offer = payload.get("offer")
                                logger.info(f"[Task] Handling WebRTC offer for {bid}")
                                if bid not in browser_manager:
                                    logger.warning(f"[Task] Browser {bid} not found for WebRTC offer")
                                    result = {"error": "Browser not found"}
                                else:
                                    try:
                                        answer = await handle_webrtc_offer_internal(bid, offer)
                                        logger.info(f"[Task] WebRTC answer generated for {bid}")
                                        result = {"answer": answer}
                                    except Exception as rtc_e:
                                        logger.error(f"[Task] WebRTC offer handling failed for {bid}: {rtc_e}")
                                        result = {"error": str(rtc_e)}
                            elif command == "rtc_ice":
                                # WebRTC signaling: handle ICE candidate from UI through server
                                bid = payload.get("browser_id")
                                candidate = payload.get("candidate")
                                if bid in browser_manager:
                                    await handle_ice_candidate_internal(bid, candidate)
                                result = {"status": "ok"}
                        except Exception as te:
                            logger.error(f"Task Execution Error: {te}")
                            health_monitor.record_error(str(te), f"task_{command}")
                            result = {"status": "error", "message": str(te)}
                        
                        async with ws_lock:
                            await ws.send(json.dumps({
                                "task_id": task_id,
                                "result": result
                            }))
                    except Exception as e:
                        logger.error(f"Message handling error: {e}")
                        health_monitor.record_error(str(e), "message_handling")

                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=60)
                        task_limiter.create_limited_task(handle_message(msg))
                    except asyncio.TimeoutError:
                        # No message for a while — send a ping by performing a noop
                        try:
                            await ws.ping()
                        except Exception:
                            # ping failed; break to reconnect
                            raise
                    except websockets.ConnectionClosed:
                        raise
                    except Exception as e:
                        # any other error should trigger reconnect
                        raise
                    
        except Exception as e:
            command_ws = None
            logger.error(f"Command channel error: {e}")
            health_monitor.record_error(str(e), "command_channel")
            node_status = "reconnecting"
            node_error = str(e)
            # Record the failure into hub_connection (computes backoff)
            try:
                hub_connection.record_failure(str(e))
                reconnect_delay = hub_connection.backoff_delay
            except Exception:
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

            # Add jitter to avoid thundering herd
            jitter = reconnect_delay * 0.25 * (np.random.random() - 0.5) * 2
            sleep_for = max(0.5, reconnect_delay + jitter)
            logger.info(f"Reconnecting in {sleep_for:.1f}s")
            await asyncio.sleep(sleep_for)

async def heartbeat_loop():
    """Send periodic heartbeats to the server"""
    global node_status, node_error, last_browser_ids
    while True:
        try:
            current_browser_ids = set(browser_manager.keys())
            should_send_list = (current_browser_ids != last_browser_ids)
            
            payload = {
                "node_id": Config.NODE_ID,
                "url": f"http://{Config.NODE_HOST}:{Config.NODE_PORT}",
                "browsers_count": len(browser_manager),
                "status": node_status,
                "error": node_error,
                "health": health_monitor.get_health_status(),
                "token": os.getenv('NODE_TOKEN')
            }
            
            if should_send_list:
                browser_list = []
                for b in browser_manager.values():
                    current_url = "about:blank"
                    if b.page:
                        try:
                            current_url = b.page.url
                        except:
                            pass
                    browser_list.append({
                        "id": b.id, 
                        "mode": b.mode, 
                        "profile_id": b.profile_id, 
                        "owner": b.owner,
                        "url": current_url
                    })
                payload["browsers"] = browser_list
                last_browser_ids = current_browser_ids

            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{Config.SERVER_URL}/register", json=payload)
                node_status = "healthy"
                node_error = None
                
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
            health_monitor.record_error(str(e), "heartbeat")
            node_status = "error"
            node_error = str(e)
            
        await asyncio.sleep(Config.HEARTBEAT_INTERVAL)

# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.post("/create")
async def create_browser(request: Request):
    """Create a new browser instance"""
    try:
        try:
            data = await request.json()
        except:
            data = {}
        
        result = await browser_manager.create(
            data.get("mode", "ephemeral"),
            data.get("profile_id"),
            data.get("owner", "system")
        )
        return result
        
    except Exception as e:
        logger.error(f"API Create Error: {e}")
        health_monitor.record_error(str(e), "api_create")
        return {"status": "error", "message": str(e)}

@app.post("/refresh")
async def node_refresh():
    """Manually clear node errors and force registration"""
    global node_status, node_error
    node_status = "healthy"
    node_error = None
    return {"status": "ok"}

@app.get("/health")
async def get_health():
    """Get node health status"""
    return health_monitor.get_health_status()

@app.post("/restart")
async def trigger_restart():
    """Trigger a graceful restart of the node"""
    asyncio.create_task(supervisor.shutdown_and_restart())
    return {"status": "ok", "message": "Restarting node..."}

@app.get("/profiles")
async def list_profiles():
    """List all stored profiles and their status"""
    if not os.path.exists(PROFILES_DIR):
        return {"profiles": []}
    
    active_profiles = {inst.profile_id: inst.id for inst in browser_manager.values() if inst.mode == "persistent"}
    
    results = []
    for d in os.listdir(PROFILES_DIR):
        path = os.path.join(PROFILES_DIR, d)
        if os.path.isdir(path):
            try:
                size = sum(f.stat().st_size for f in os.scandir(path) if f.is_file())
            except:
                size = 0
            results.append({
                "profile_id": d,
                "is_active": d in active_profiles,
                "browser_id": active_profiles.get(d),
                "size": size
            })
    return {"profiles": results}

@app.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    """Delete a profile if it's not active"""
    active_profiles = {inst.profile_id for inst in browser_manager.values() if inst.mode == "persistent"}
    if profile_id in active_profiles:
        return {"status": "error", "message": "Cannot delete an active profile"}
    
    import shutil
    path = os.path.join(PROFILES_DIR, profile_id)
    if os.path.exists(path):
        shutil.rmtree(path)
        return {"status": "ok"}
    return {"status": "error", "message": "Profile not found"}

@app.post("/close/{bid}")
async def close_browser(bid: str):
    """Close a browser instance"""
    return await browser_manager.close(bid)

@app.post("/api/offer/{bid}")
async def webrtc_offer(bid: str, request: Request):
    """Handle WebRTC offer"""
    if bid not in browser_manager:
        return {"error": "not_found"}
    
    logger.info(f"WebRTC Offer for {bid}")
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    pc = None
    track = None
    
    try:
        pc = RTCPeerConnection(configuration=RTCConfiguration(
            iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        ))
        browser_manager[bid].pcs.append(pc)
        
        track = BrowserVideoTrack()
        browser_manager[bid].tracks.append(track)
        
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
                    task_limiter.create_limited_task(browser_manager[bid].handle_input(action))
                except Exception as e:
                    logger.error(f"DC Parse Error: {e}")

        @pc.on("connectionstatechange")
        async def on_state():
            logger.info(f"PC State {bid}: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                if bid in browser_manager:
                    if pc in browser_manager[bid].pcs:
                        browser_manager[bid].pcs.remove(pc)
                    if track in browser_manager[bid].tracks:
                        browser_manager[bid].tracks.remove(track)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        
    except Exception as e:
        logger.error(f"WebRTC Handshake Failed: {e}")
        health_monitor.record_error(str(e), f"webrtc_{bid}")
        
        try:
            if bid in browser_manager:
                if pc and pc in browser_manager[bid].pcs:
                    browser_manager[bid].pcs.remove(pc)
                if track and track in browser_manager[bid].tracks:
                    browser_manager[bid].tracks.remove(track)
        except:
            pass
        
        return {"error": str(e)}

@app.websocket("/ws/{bid}")
async def browser_ws(websocket: WebSocket, bid: str):
    """WebSocket endpoint for browser streaming"""
    if bid not in browser_manager:
        await websocket.close()
        return
    
    await websocket.accept()
    instance = browser_manager[bid]
    instance.websockets.append(websocket)
    logger.info(f"WS Connected {bid}")
    
    try:
        while True:
            data = await websocket.receive_text()
            action = json.loads(data)
            task_limiter.create_limited_task(instance.handle_input(action))
    except WebSocketDisconnect:
        logger.warning(f"WS Disconnected {bid}")
    except Exception as e:
        logger.error(f"WS Error {bid}: {e}")
    finally:
        if websocket in instance.websockets:
            instance.websockets.remove(websocket)

# =============================================================================
# AUTO-RESTART / SELF-HEALING
# =============================================================================
class NodeSupervisor:
    """Supervises the node and handles auto-restart"""
    
    def __init__(self):
        self.restart_count = 0
        self.last_restart = 0
        self._should_restart = False
        self.loop = None
        
    def check_health_and_restart(self):
        """Check if node should restart based on health"""
        if not Config.AUTO_RESTART_ON_CRASH:
            return False
        
        if self.restart_count >= Config.MAX_RESTART_ATTEMPTS:
            logger.error("Max restart attempts reached, not restarting")
            return False
        
        if time.time() - self.last_restart < Config.RESTART_COOLDOWN:
            return False
        
        if health_monitor.should_restart():
            logger.warning("Health check failed, triggering restart")
            self._should_restart = True
            return True
        
        return False
    
    async def shutdown_and_restart(self):
        """Perform a clean shutdown of all resources and restart the process"""
        logger.info("Graceful restart initiated - cleaning up all resources...")
        
        # 1. Stop the browser manager (closes all playwright instances)
        try:
            await browser_manager.close_all()
        except Exception as e:
            logger.error(f"Error closing browsers during restart: {e}")
            
        # 2. Shutdown pools
        try:
            pool_manager.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error shutting down pools during restart: {e}")
            
        # 3. Small delay to ensure OS handles release
        await asyncio.sleep(1)
        
        # 4. Re-execute the current script
        logger.info("Executing process restart in 2 seconds...")
        python = sys.executable
        
        # Filter existing --wait from args
        clean_args = []
        skip = False
        for a in sys.argv[1:]:
            if skip: skip = False; continue
            if a == '--wait': skip = True; continue
            if a.startswith('--wait='): continue
            clean_args.append(a)
        
        restart_args = clean_args + ['--wait', '2']

        if os.name == 'nt':
            try:
                # Use subprocess.Popen with CREATE_NEW_CONSOLE for Windows
                subprocess.Popen([python] + restart_args, creationflags=subprocess.CREATE_NEW_CONSOLE)
                os._exit(0)
            except Exception as e:
                logger.error(f"Failed to restart via Popen: {e}")
                os.execl(python, python, *restart_args)
        else:
            os.execl(python, python, *restart_args)
            
    def restart_node(self):
        """Restart the node process (synchronous entry point)"""
        # If we're already shutting down, don't trigger again
        if hasattr(self, '_restarting') and self._restarting:
            return
            
        self._restarting = True
        self.restart_count += 1
        self.last_restart = time.time()
        health_monitor.record_restart()
        
        logger.info(f"Restarting node (attempt {self.restart_count}/{Config.MAX_RESTART_ATTEMPTS})")
        
        # Try to use the running event loop to perform graceful shutdown
        try:
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.shutdown_and_restart(), self.loop)
                return
        except Exception:
            pass
            
        # Fallback to immediate exec if no loop is available (less clean but works)
        python = sys.executable
        
        # Build restart command
        if getattr(sys, 'frozen', False):
            # Running as exe - sys.executable IS the exe
            restart_cmd = [python, '--wait', '2']
        else:
            # Running as script
            restart_cmd = [python, sys.argv[0], '--wait', '2']
        
        if os.name == 'nt':
            try:
                subprocess.Popen(restart_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
                os._exit(0)
            except Exception as e:
                logger.error(f"Fallback restart failed: {e}")
                os.execl(restart_cmd[0], *restart_cmd)
        else:
            os.execl(restart_cmd[0], *restart_cmd)

supervisor = NodeSupervisor()

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down...")
    
    # Run cleanup in event loop
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(browser_manager.close_all())
    
    pool_manager.shutdown(wait=False)
    sys.exit(0)

# =============================================================================
# SYSTEM TRAY MANAGER
# =============================================================================
class TrayIconManager:
    """Manages the system tray icon for Windows"""
    
    def __init__(self):
        self.icon = None
        self._thread = None
        self._running = False
        
    def _create_image(self):
        """Generate a simple dynamic icon for StealthNode"""
        # Create a 64x64 image with a dark background
        width = 64
        height = 64
        image = Image.new('RGB', (width, height), color=(15, 15, 20))
        dc = ImageDraw.Draw(image)
        
        # Draw a stylized 'SN' with a glow effect
        # Simple blue/teal circle
        dc.ellipse([8, 8, 56, 56], fill=(45, 212, 191), outline=(255, 255, 255))
        
        # Center the text 'SN' (Black color)
        # Using default font which is safe for cross-platform
        dc.text((18, 18), "SN", fill=(15, 15, 20))
        
        return image

    def _on_restart(self, icon, item):
        logger.info("Restart requested from tray icon")
        
        try:
            # Get the actual executable path
            if getattr(sys, 'frozen', False):
                # Running as compiled exe - sys.executable IS the exe path
                exe_path = sys.executable
                exe_dir = os.path.dirname(exe_path)
            else:
                # Running as script
                exe_path = sys.executable  # python.exe
                script_path = os.path.abspath(sys.argv[0])
                exe_dir = os.path.dirname(script_path)
            
            logger.info(f"Executable path: {exe_path}")
            logger.info(f"Executable dir: {exe_dir}")
            
            # Create batch file in the same directory as the exe (for debugging visibility)
            batch_path = os.path.join(exe_dir, 'restart_node.bat')
            
            if getattr(sys, 'frozen', False):
                # For frozen exe, just restart the exe
                batch_content = f'''@echo off
echo Ensuring all Node processes are terminated...
taskkill /F /IM Node.exe /T > nul 2>&1
echo Waiting 10 seconds for complete resource release...
timeout /t 10 /nobreak > nul
echo Starting Node...
start "" "{exe_path}"
echo Deleting this batch file...
(goto) 2>nul & del "%~f0"
'''
            else:
                # For script, run python with the script
                script_path = os.path.abspath(sys.argv[0])
                batch_content = f'''@echo off
echo Ensuring all Python Node processes are terminated...
taskkill /F /FI "WINDOWTITLE eq StealthNode*" /T > nul 2>&1
echo Waiting 10 seconds for complete resource release...
timeout /t 10 /nobreak > nul
echo Starting Node...
"{exe_path}" "{script_path}"
echo Deleting this batch file...
(goto) 2>nul & del "%~f0"
'''
            
            # Write batch file
            logger.info(f"Writing batch file to: {batch_path}")
            with open(batch_path, 'w') as f:
                f.write(batch_content)
            
            logger.info(f"Batch content:\n{batch_content}")
            
            # Run the batch file in a new console window
            subprocess.Popen(
                f'cmd /c "{batch_path}"',
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            logger.info("Batch file started in new console, exiting current process...")
            
            # Stop tray icon
            if self.icon:
                self.icon.stop()
            
            # Exit immediately
            os._exit(0)
            
        except Exception as e:
            logger.error(f"Restart failed: {e}")
            import traceback
            traceback.print_exc()

    def _on_exit(self, icon, item):
        logger.info("Exit requested from tray icon")
        icon.stop()
        # Trigger graceful shutdown signals
        os.kill(os.getpid(), signal.SIGTERM)

    def _on_open_hub(self, icon, item):
        """Open the server hub in the default browser"""
        hub_url = Config.SERVER_URL
        webbrowser.open(hub_url)

    def _on_view_logs(self, icon, item):
        """Open the log folder in explorer"""
        if os.path.exists(LOGS_DIR):
            if platform.system() == "Windows":
                os.startfile(LOGS_DIR)
            else:
                subprocess.Popen(['open', LOGS_DIR] if platform.system() == 'Darwin' else ['xdg-open', LOGS_DIR])

    def run(self):
        """Initialize and run the tray icon in its own event loop"""
        if pystray is None:
            logger.warning("pystray not installed, system tray icon disabled")
            return

        try:
            menu = (
                item('Open Command Center', self._on_open_hub, default=True),
                item('View Logs', self._on_view_logs),
                item('Restart Node', self._on_restart),
                pystray.Menu.SEPARATOR,
                item('Exit', self._on_exit)
            )
            
            self.icon = pystray.Icon(
                "StealthNode",
                self._create_image(),
                "StealthNode Node Engine",
                menu
            )
            
            logger.info("System tray icon initialized")
            self.icon.run() # This blocks its thread
        except Exception as e:
            logger.error(f"Tray icon error: {e}")

    def start(self):
        """Start the tray icon in a background thread"""
        if self._thread is not None:
            return
            
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the tray icon"""
        if self.icon:
            self.icon.stop()

tray_manager = TrayIconManager()
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def find_available_port(start_port: int, host: str = "0.0.0.0") -> int:
    """Find the first available port starting from start_port"""
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                port += 1
    return start_port

def main():
    """Main entry point for the node"""
    # Quick check for --wait flag before anything else
    for i, arg in enumerate(sys.argv):
        if arg == '--wait' and i + 1 < len(sys.argv):
            try:
                wait_time = int(sys.argv[i+1])
                time.sleep(wait_time)
            except: pass

    # Startup registration
    def check_startup():
        """Add the application to Windows startup if not already present"""
        if platform.system() != "Windows":
            return
        try:
            import winreg
            # Get actual target path
            if getattr(sys, 'frozen', False):
                # If compiled EXE, use its path directly
                target_path = sys.executable
            else:
                # If running as script, use pythonw and the script path
                exe = sys.executable.replace("python.exe", "pythonw.exe")
                target_path = f'"{exe}" "{os.path.abspath(sys.argv[0])}"'
                
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
            # We use HKCU to avoid requiring Administrator permissions
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_ALL_ACCESS) as key:
                try:
                    val, _ = winreg.QueryValueEx(key, "StealthNode")
                    if val == target_path:
                        return # Already set correctly
                except FileNotFoundError:
                    pass
                
                winreg.SetValueEx(key, "StealthNode", 0, winreg.REG_SZ, target_path)
                logger.info(f"StealthNode registered to startup: {target_path}")
        except Exception as e:
            logger.error(f"Startup registration failed: {e}")

    check_startup()
    
    # Fix for compiled EXE without console: ensure stdout/stderr are not None
    if sys.stdout is None:
        class NullWriter:
            def write(self, *args, **kwargs): pass
            def flush(self): pass
            def isatty(self): return False
        sys.stdout = NullWriter()
    if sys.stderr is None:
        class NullWriter:
            def write(self, *args, **kwargs): pass
            def flush(self): pass
            def isatty(self): return False
        sys.stderr = NullWriter()

    # Ensure browser is available (Portable Mode fallback)
    def ensure_browser():
        if Config.get_browser_channel() == "chrome":
            logger.info("System Chrome detected. Skipping portable browser installation.")
            return

        try:
            import subprocess
            logger.info("Chrome not found. Ensuring portable Chromium is available...")
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], capture_output=True)
            logger.info("Portable browser verified.")
        except Exception as e:
            logger.error(f"Failed to verify/install portable browser: {e}")

    ensure_browser()
    logger.info("=" * 60)
    logger.info("STEALTH NODE - ULTRA HIGH-PERFORMANCE BROWSER ENGINE")
    logger.info("=" * 60)
    logger.info(f"Node ID: {Config.NODE_ID}")
    logger.info(f"Server URL: {Config.SERVER_URL}")
    logger.info(f"Max Browsers: {Config.MAX_BROWSERS}")
    logger.info(f"Thread Pool Size: {Config.THREAD_POOL_SIZE}")
    logger.info(f"Process Pool Size: {Config.PROCESS_POOL_SIZE}")
    logger.info(f"Memory Limit: {Config.MEMORY_LIMIT_MB}MB")
    logger.info("=" * 60)
    
    # Start System Tray Icon
    tray_manager.start()
    
    # Start periodic health check in background
    def health_check_thread():
        while True:
            time.sleep(60)
            if supervisor.check_health_and_restart():
                supervisor.restart_node()
    
    health_thread = threading.Thread(target=health_check_thread, daemon=True)
    health_thread.start()
    
    # Dynamic Port Discovery to allow multiple nodes on same host
    original_port = Config.NODE_PORT
    Config.NODE_PORT = find_available_port(original_port)
    
    if Config.NODE_PORT != original_port:
        logger.info(f"Port {original_port} was busy. Dynamic Discovery selected port {Config.NODE_PORT}")

    # Run the server
    # use_colors must be False if stdout is None to avoid crash in compiled EXE
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.NODE_PORT,
        log_level="info",
        access_log=False,  # Disable access log for performance
        workers=1,  # Single worker, async handles concurrency
        limit_concurrency=10000,  # High concurrency limit
        limit_max_requests=None,  # No request limit
        timeout_keep_alive=30,
        use_colors=sys.stdout.isatty() if (sys.stdout and hasattr(sys.stdout, 'isatty')) else False
    )

if __name__ == "__main__":
    # PyInstaller compatibility
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        multiprocessing.freeze_support()
    
    main()
