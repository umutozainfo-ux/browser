import asyncio
import logging
import socketio
import uuid
import time
import os
import base64
import av
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp
from browser import BrowserManager
from fractions import Fraction
import traceback

# Set log levels to reduce noise but keep critical info
logging.basicConfig(level=logging.INFO)
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)
logger = logging.getLogger("Node")

# Global State
NODE_ID = f"node-{str(uuid.uuid4())[:8]}"
browser = BrowserManager()
sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=0, reconnection_delay=1, reconnection_delay_max=30)
pcs = set()
fallback_active = False
fallback_task = None
encoder = None
force_next_keyframe = False
current_proxy = None
current_proxy_info = "Direct"
browser_lock = asyncio.Lock()

# Video Track that yields Playwright frames (WebRTC)
class BrowserVideoTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, browser_manager):
        super().__init__()
        self.browser_manager = browser_manager
    async def recv(self):
        # Optimized: get_latest_frame now waits for the event, no polling needed
        frame = await self.browser_manager.get_latest_frame()
        return frame

# H.264 Encoder for WebSocket Fallback
class H264Encoder:
    def __init__(self, width=1280, height=720, fps=60):
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = None
        self.frame_index = 0
        self._init_codec()
    
    def _init_codec(self):
        try:
            self.codec = av.CodecContext.create('h264', 'w')
            self.codec.width = self.width
            self.codec.height = self.height
            self.codec.pix_fmt = 'yuv420p'
            self.codec.framerate = self.fps
            self.codec.time_base = Fraction(1, 1000) # Use ms timebase
            self.codec.options = {
                'preset': 'ultrafast',
                'tune': 'zerolatency',
                'profile': 'baseline',
                'crf': '28',
                'x264opts': 'keyint=120:min-keyint=120:repeat-headers=1:sliced-threads=0:sync-lookahead=0:rc-lookahead=0'
            }
            self.codec.open()
        except Exception as e:
            logger.error(f"Failed to init H.264 codec: {e}")

    def encode(self, frame_bgr, force_keyframe=False):
        if self.codec is None: return None
        try:
            frame = av.VideoFrame.from_ndarray(frame_bgr, format='bgr24')
            
            # Simple monotonic PTS
            self.frame_index += 1
            frame.pts = self.frame_index
            frame.time_base = self.codec.time_base
            
            if force_keyframe:
                frame.key_frame = True
            
            packets = self.codec.encode(frame)
            data = b''
            for packet in packets:
                # PyAV usually includes Annex B for H264 if headers are repeated, 
                # but we ensure binary stability here.
                data += bytes(packet)
            
            if data and not data.startswith(b'\x00\x00\x00\x01') and not data.startswith(b'\x00\x00\x01'):
                # If for some reason it's not Annex B (e.g. avcC), this is a problem for JMuxer.
                # But x264 usually produces Annex B. 
                pass

            return data
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            logger.error(traceback.format_exc())
            return None

async def start_browser_safe():
    """Starts browser with locking to prevent double initialization"""
    global current_proxy, current_proxy_info
    async with browser_lock:
        if browser.is_healthy(): return
        logger.info(f"Initializing Browser ({current_proxy_info})...")
        try:
            await browser.start(proxy=current_proxy)
        except Exception as e:
            if "PROXY_FAILURE" in str(e):
                logger.warning("Proxy failed, falling back to Direct...")
                current_proxy = None
                current_proxy_info = "Direct (Fallback)"
                await browser.start(proxy=None)
            else:
                raise e

@sio.event
async def connect():
    logger.info(f"Connected to Server as {NODE_ID}")
    # Registration will be handled by main loop or here
    await sio.emit('register_browser', {
        'id': NODE_ID, 
        'proxy_info': current_proxy_info, 
        'capabilities': ['webrtc', 'fallback_h264']
    })
    # Start streaming immediately if requested
    if fallback_active:
        await request_fallback({})

@sio.event
async def disconnect():
    logger.info("Disconnected from Server")

@sio.event
async def request_fallback(data):
    """Client requested fallback mode"""
    global fallback_active, fallback_task, encoder
    logger.info("Starting Fallback H.264 Stream...")
    
    # Ensure browser is running
    await start_browser_safe()
    
    # (Re)init encoder
    encoder = H264Encoder() 
    fallback_active = True
    
    if fallback_task is None or fallback_task.done():
        fallback_task = asyncio.create_task(run_fallback_loop())

@sio.event
async def stop_fallback(data):
    global fallback_active
    logger.info("Stopping Fallback Stream...")
    fallback_active = False

@sio.event
async def request_keyframe(data):
    global force_next_keyframe
    logger.debug("Keyframe requested (Healing)...")
    force_next_keyframe = True
    browser.force_refresh() # Fix: Ensure next frame is processed

MODE_MJPEG = False

@sio.event
async def enable_mjpeg(data):
    global MODE_MJPEG
    MODE_MJPEG = data.get('enabled', False)
    logger.info(f"MJPEG Mode set to: {MODE_MJPEG}")
    browser.force_refresh() # Fix: Ensure immediate update on switch

async def run_fallback_loop():
    global fallback_active, encoder, force_next_keyframe, MODE_MJPEG
    logger.info("Fallback encoded loop started.")
    
    last_frame_pts = 0
    
    while True:
        if not fallback_active:
            await asyncio.sleep(0.5)
            continue
            
        try:
            # Wait for next frame (event driven)
            await browser.frame_event.wait()
            
            if MODE_MJPEG:
                # MJPEG Path: Ultra Low Latency, High Bandwidth
                jpeg_bytes = browser.get_latest_jpeg()
                if jpeg_bytes:
                     await sio.emit('mjpeg_data', jpeg_bytes)
            else:
                # H.264 Path: Low Bandwidth, Higher Latency
                img, pts = browser.get_latest_raw()
                if img is not None:
                    if pts != last_frame_pts:
                        last_frame_pts = pts
                        data = await asyncio.get_event_loop().run_in_executor(None, encoder.encode, img, force_next_keyframe)
                        
                        if force_next_keyframe: 
                            force_next_keyframe = False
                            logger.info("IDR Frame Generated")

                        if data:
                            await sio.emit('video_data', data)
                
        except Exception as e:
            logger.error(f"Fallback loop error: {e}")
            await asyncio.sleep(0.1)

@sio.event
async def offer(data):
    """WebRTC Offer Handler"""
    await start_browser_safe()
    
    config = RTCConfiguration(iceServers=[RTCIceServer(urls='stun:stun.l.google.com:19302')])
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)

    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
         if candidate:
            await sio.emit('ice_candidate', {
                'target': data['from'],
                'candidate': {'candidate': candidate.candidate, 'sdpMid': candidate.sdpMid, 'sdpMLineIndex': candidate.sdpMLineIndex}
            })

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed"]:
            pcs.discard(pc)

    try:
        pc.addTrack(BrowserVideoTrack(browser))
        await pc.setRemoteDescription(RTCSessionDescription(sdp=data['sdp'], type=data['type']))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await sio.emit('answer', {'target': data['from'], 'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})
    except Exception as e:
        logger.error(f"WebRTC Error: {e}")
        await pc.close()
        pcs.discard(pc)

@sio.event
async def ice_candidate(data):
    cand_dict = data.get('candidate')
    if not cand_dict: return
    try:
        candidate = candidate_from_sdp(cand_dict['candidate'].split(':', 1)[1] if 'candidate:' in cand_dict['candidate'] else cand_dict['candidate'])
        candidate.sdpMid = cand_dict.get('sdpMid')
        candidate.sdpMLineIndex = cand_dict.get('sdpMLineIndex')
        for pc in list(pcs):
            if pc.connectionState not in ["closed", "failed"]:
                await pc.addIceCandidate(candidate)
    except Exception as e:
        logger.error(f"ICE candidate error: {e}")

@sio.event
async def control_event(data):
    try: await browser.handle_input(data)
    except: pass

@sio.on('restart_browser')
async def restart_browser(data):
    logger.info("Restarting browser per request...")
    async with browser_lock:
        for pc in list(pcs): await pc.close()
        pcs.clear()
        await browser.close()
        await browser.start(proxy=current_proxy)

@sio.on('set_proxy')
async def set_proxy(data):
    global current_proxy, current_proxy_info
    async with browser_lock:
        new_proxy = data.get('proxy')
        current_proxy = new_proxy
        current_proxy_info = f"Proxy: {new_proxy.get('server', 'Unknown')}" if new_proxy else "Direct"
        for pc in list(pcs): await pc.close()
        pcs.clear()
        await browser.close()
        await browser.start(proxy=current_proxy)

async def main():
    global current_proxy, current_proxy_info
    server_url = os.getenv('SERVER_URL', 'http://localhost:5000')
    
    while True:
        try:
            if not sio.connected:
                await sio.connect(server_url, wait_timeout=10)
            
            if current_proxy is None:
                try:
                    current_proxy = await sio.call('request_proxy', {}, timeout=5)
                    current_proxy_info = f"Proxy: {current_proxy.get('server', 'Unknown')}" if current_proxy else "Direct"
                except:
                    current_proxy_info = "Direct"

            await start_browser_safe()
            
            while True:
                await asyncio.sleep(5)
                if not sio.connected: break
                if not browser.is_healthy(): break
        except Exception as e:
            logger.error(f"Node Main Error: {e}")
            await asyncio.sleep(5)

if __name__ == '__main__':
    asyncio.run(main())
