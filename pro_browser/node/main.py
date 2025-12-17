import asyncio
import logging
import socketio
import uuid
import time
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, MediaStreamTrack
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
        # Limit frame rate slightly to avoid CPU starvation?
        # CDP pushes frames, but we pull here.
        # Ideally we wait for a NEW frame event.
        
        # Simple polling for now
        start_time = time.time()
        while True:
            frame = await self.browser_manager.get_latest_frame()
            if frame:
                # Calculate proper timestamp
                # Monotonically increasing PTS based on time
                # Re-calculate to ensure it's current for the stream even if frame is old
                pts = int(time.time() * 90000)
                frame.pts = pts
                from fractions import Fraction
                frame.time_base = Fraction(1, 90000)
                return frame
            
            await asyncio.sleep(0.01)
            if time.time() - start_time > 1.0:
                # Timeout, return black frame? or just wait
                pass

sio = socketio.AsyncClient()
browser = BrowserManager()
pcs = set()

# Random Node ID
NODE_ID = f"node-{str(uuid.uuid4())[:8]}"

@sio.event
async def connect():
    logger.info(f"Connected to Server as {NODE_ID}")
    await sio.emit('register_browser', {'id': NODE_ID})

@sio.event
async def offer(data):
    logger.info(f"Received WebRTC Offer from {data['from']}")
    
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
         if candidate:
            # We construct the same object structure that socket.io expects on the other side
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

    # Add Video Track
    pc.addTrack(BrowserVideoTrack(browser))

    # Set Remote Description
    await pc.setRemoteDescription(RTCSessionDescription(sdp=data['sdp'], type=data['type']))
    
    # Create Answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    await sio.emit('answer', {
        'target': data['from'],
        'sdp': pc.localDescription.sdp,
        'type': pc.localDescription.type
    })

@sio.event
async def ice_candidate(data):
    cand_dict = data['candidate']
    # Use aiortc.sdp.candidate_from_sdp to parse the candidate string
    candidate_str = cand_dict['candidate']
    sdp_mid = cand_dict.get('sdpMid')
    sdp_mline_index = cand_dict.get('sdpMLineIndex')
    
    # candidate_from_sdp only parses the candidate string into an object, 
    # but we need to assign mid/index
    # The string format is: 'candidate:...'
    # Wait, aiortc might expect us to split it?
    # candidate_from_sdp handles the "candidate:..." part?
    # Actually, a simpler way is often just letting the library handle it or check docs.
    # The error was RTCIceCandidate() got unexpected keyword 'candidate'.
    # This means I constructed it wrong.
    
    # Correct way using candidate_from_sdp:
    # cand = candidate_from_sdp(candidate_str)
    # cand.sdpMid = sdp_mid
    # cand.sdpMLineIndex = sdp_mline_index
    
    if 'candidate:' in candidate_str:
        candidate = candidate_from_sdp(candidate_str.split(':', 1)[1])
    else:
        candidate = candidate_from_sdp(candidate_str)
        
    candidate.sdpMid = sdp_mid
    candidate.sdpMLineIndex = sdp_mline_index
    
    for pc in pcs:
        await pc.addIceCandidate(candidate)

@sio.event
async def control_event(data):
    await browser.handle_input(data)

async def main():
    await browser.start()
    
    import os
    server_url = os.getenv('SERVER_URL', 'http://localhost:5000')
    try:
        # Connect to Server (assuming docker service name 'server' or localhost)
        await sio.connect(server_url)
        await sio.wait()
    finally:
        await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
