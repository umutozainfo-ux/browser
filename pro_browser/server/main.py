import logging
import socketio
from quart import Quart, render_template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Server")

app = Quart(__name__)
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app.asgi_app = socketio.ASGIApp(sio, app.asgi_app)

# Store connected browsers and controllers
browsers = {}

@app.route('/')
async def index():
    return await render_template('index.html')

@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")
    # Immediately sync current browsers to the new client (UI)
    await sio.emit('browser_list_update', list(browsers.values()), room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    if sid in browsers:
        del browsers[sid]
        # Notify controllers that a browser left
        await sio.emit('browser_list_update', list(browsers.values()))

@sio.event
async def register_browser(sid, data):
    """Register a new browser node"""
    browser_id = data.get('id', sid)
    browsers[sid] = {'id': browser_id, 'sid': sid}
    logger.info(f"Browser registered: {browser_id} (SID: {sid})")
    await sio.emit('browser_list_update', list(browsers.values()))

@sio.event
async def offer(sid, data):
    """Relay WebRTC offer from Browser to Controller (or vice versa)"""
    target = data.get('target')
    if target:
        await sio.emit('offer', {'sdp': data['sdp'], 'type': data['type'], 'from': sid}, room=target)

@sio.event
async def answer(sid, data):
    """Relay WebRTC answer"""
    target = data.get('target')
    if target:
        await sio.emit('answer', {'sdp': data['sdp'], 'type': data['type'], 'from': sid}, room=target)

@sio.event
async def ice_candidate(sid, data):
    """Relay ICE candidate"""
    target = data.get('target')
    if target:
        await sio.emit('ice_candidate', {'candidate': data['candidate'], 'from': sid}, room=target)

# Control events
@sio.event
async def control_event(sid, data):
    """Relay control input (mouse/keyboard) to target browser(s)"""
    targets = data.get('target')
    if targets:
        if isinstance(targets, list):
            for t in targets:
                await sio.emit('control_event', data, room=t)
        else:
            await sio.emit('control_event', data, room=targets)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
