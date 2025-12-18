import asyncio
import logging
import socketio
import uuid
import time
import os
import base64
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp
from browser import BrowserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Node")

# Video Track that yields Playwright frames
class BrowserVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, browser_manager):
        super().__init__()
        self.browser_manager = browser_manager

    async def recv(self):
        last_log = time.time()
        while True:
            frame = await self.browser_manager.get_latest_frame()
            if frame:
                return frame
            
            if time.time() - last_log > 10:
                logger.warning("No frames received from browser for 10s...")
                last_log = time.time()
                
            await asyncio.sleep(0.1)

sio = socketio.AsyncClient()
browser = BrowserManager()
pcs = set()
offer_lock = asyncio.Lock()

# Random Node ID
NODE_ID = f"node-{str(uuid.uuid4())[:8]}"

@sio.event
async def connect():
    logger.info(f"Connected to Server as {NODE_ID}")
    try:
        await sio.emit('register_browser', {'id': NODE_ID})
    except Exception as e:
        logger.error(f"Error during registration: {e}")

@sio.event
async def disconnect():
    logger.info("Disconnected from Server")

@sio.event
async def connect_error(data):
    logger.error(f"Connection Error: {data}")

@sio.event
async def offer(data):
    """Handle WebRTC Offer with serialization and health wait"""
    async with offer_lock:
        logger.info(f"Processing WebRTC Offer from {data.get('from')}...")
        
        # Configure ICE Servers (STUN/TURN)
        ice_servers = [
            RTCIceServer(urls=[
                "stun:stun.l.google.com:19302",
                "stun:stun1.l.google.com:19302",
                "stun:stun2.l.google.com:19302",
                "stun:stun.services.mozilla.com"
            ])
        ]
        
        turn_url = os.getenv('TURN_URL')
        if turn_url:
            ice_servers.append(RTCIceServer(
                urls=[turn_url],
                username=os.getenv('TURN_USERNAME'),
                credential=os.getenv('TURN_PASSWORD')
            ))

        config = RTCConfiguration(iceServers=ice_servers)
        pc = RTCPeerConnection(configuration=config)
        
        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
             if candidate:
                await sio.emit('ice_candidate', {
                    'target': data['from'],
                    'candidate': {
                        'candidate': candidate.candidate, 
                        'sdpMid': candidate.sdpMid, 
                        'sdpMLineIndex': candidate.sdpMLineIndex
                    }
                })

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        # Wait for browser to be healthy
        wait_start = time.time()
        while not browser.is_healthy():
            if time.time() - wait_start > 30:
                logger.error("Offer timed out waiting for browser health")
                await pc.close()
                return
            if pc.connectionState == "closed":
                return
            await asyncio.sleep(0.5)

        if pc.connectionState == "closed":
            return

        pcs.add(pc)

        try:
            pc.addTrack(BrowserVideoTrack(browser))
            await pc.setRemoteDescription(RTCSessionDescription(sdp=data['sdp'], type=data['type']))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await sio.emit('answer', {
                'target': data['from'],
                'sdp': pc.localDescription.sdp,
                'type': pc.localDescription.type
            })
            logger.info(f"Handshake complete for {data['from']}")
        except Exception as e:
            if pc.connectionState != "closed":
                logger.error(f"Error during offer handling: {e}")
            await pc.close()
            pcs.discard(pc)

@sio.event
async def ice_candidate(data):
    cand_dict = data.get('candidate')
    if not cand_dict: return
    candidate_str = cand_dict.get('candidate')
    sdp_mid = cand_dict.get('sdpMid')
    sdp_mline_index = cand_dict.get('sdpMLineIndex')
    
    try:
        if 'candidate:' in candidate_str:
            candidate = candidate_from_sdp(candidate_str.split(':', 1)[1])
        else:
            candidate = candidate_from_sdp(candidate_str)
        candidate.sdpMid = sdp_mid
        candidate.sdpMLineIndex = sdp_mline_index
        for pc in list(pcs):
            if pc.connectionState not in ["closed", "failed"]:
                await pc.addIceCandidate(candidate)
    except Exception as e:
        logger.error(f"Error parsing ICE candidate: {e}")

@sio.event
async def control_event(data):
    try:
        etype = data.get('type')
        logger.info(f"Control event received: {etype}")
        await browser.handle_input(data)
    except Exception as e:
        logger.debug(f"Input handling suppressed: {e}")

@sio.on('restart_browser')
async def restart_browser(data):
    logger.info("Server requested browser restart...")
    for pc in list(pcs):
        await pc.close()
    pcs.clear()
    await browser.close()

async def main():
    logger.info(f"Starting Browser Node {NODE_ID}...")
    server_url = os.getenv('SERVER_URL', 'http://localhost:5000')

    while True:
        try:
            await browser.close() 
            logger.info("Initializing Browser...")
            await browser.start()
            if not sio.connected:
                logger.info(f"Connecting to server at {server_url}...")
                await sio.connect(server_url, wait_timeout=10)
            else:
                await sio.emit('register_browser', {'id': NODE_ID})
            
            while browser.is_healthy():
                await asyncio.sleep(2)
                if not sio.connected:
                    try: await sio.connect(server_url, wait_timeout=5)
                    except: pass
            logger.warning("Browser unhealthy or loop broken, restarting...")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await asyncio.sleep(5)
        finally:
            for pc in list(pcs): await pc.close()
            pcs.clear()
            await browser.close()
            await asyncio.sleep(1)

if __name__ == '__main__':
    asyncio.run(main())
