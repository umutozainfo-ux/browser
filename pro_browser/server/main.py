import logging
import asyncio
import json
import os
import datetime
import socketio
from functools import wraps
from quart import Quart, render_template, request, redirect, url_for, session, abort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Server")

app = Quart(__name__)
app.secret_key = "super-secret-grid-control-key" # For session security
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*', logger=True, engineio_logger=True)
app.asgi_app = socketio.ASGIApp(sio, app.asgi_app)

# Persistent user storage
USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')

def load_users():
    if not os.path.exists(USERS_FILE):
        return {"admin": {"password": "joepop", "role": "admin"}, "users": {}}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(data):
    with open(USERS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

PROXIES_FILE = os.path.join(os.path.dirname(__file__), 'proxies.json')

def load_proxies():
    if not os.path.exists(PROXIES_FILE):
        return []
    with open(PROXIES_FILE, 'r') as f:
        try:
            return json.load(f)
        except:
            return []

def save_proxies(proxies):
    with open(PROXIES_FILE, 'w') as f:
        json.dump(proxies, f, indent=4)

# Store connected browsers
# browsers[sid] = {'id': id, 'sid': sid, 'in_use_by': username or None, 'assigned_at': timestamp or None, 'proxy_info': str, 'is_held': bool}
browsers = {}

# Node ID -> Username persistent leases
# This survives SID changes (browser reconnects)
node_leases = {}

# Held nodes mapping: node_id -> {username, held_at, expiry}
# When a user disconnects, their nodes are NOT freed if held
held_nodes = {}
HOLD_TIMEOUT_MINUTES = 10  # Nodes are released after 10 minutes of user being offline

# List of usernames waiting for nodes (FIFO)
waiting_users = []

# sid -> username mapping for security
sid_to_user = {}

# Track last seen time for users (for hold expiry)
user_last_seen = {}

# Global lock for node allocation to prevent race conditions
allocation_lock = asyncio.Lock()

def login_required(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return await f(*args, **kwargs)
    return decorated_function

def get_stats():
    total = len(browsers)
    in_use = len([b for b in browsers.values() if b.get('in_use_by')])
    return {
        "total": total,
        "in_use": in_use,
        "free": total - in_use
    }

async def broadcast_stats():
    await sio.emit('admin_stats', get_stats(), room='role_admin')

async def broadcast_users():
    db = load_users()
    for name, user in db['users'].items():
        exp = datetime.datetime.fromisoformat(user['expires_at'])
        diff = exp - datetime.datetime.now()
        user['expires_at_human'] = "Expired" if diff.total_seconds() < 0 else f"{int(diff.total_seconds() // 3600)}h {int((diff.total_seconds() % 3600) // 60)}m"
    await sio.emit('user_list_update', db['users'], room='role_admin')

@app.route('/login', methods=['GET', 'POST'])
async def login():
    if request.method == 'POST':
        form = await request.form
        username = form.get('username')
        password = form.get('password')
        
        db = load_users()
        if username == 'admin' and db['admin']['password'] == password:
            session['user'] = 'admin'
            session['role'] = 'admin'
            return redirect(url_for('admin'))
        
        if username in db['users'] and db['users'][username]['password'] == password:
            expires_at = datetime.datetime.fromisoformat(db['users'][username]['expires_at'])
            if datetime.datetime.now() < expires_at:
                session['user'] = username
                session['role'] = 'user'
                return redirect(url_for('index'))
            else:
                return await render_template('login.html', error="Account expired")
        
        return await render_template('login.html', error="Invalid credentials")
    return await render_template('login.html')

@app.route('/logout')
async def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
async def index():
    db = load_users()
    user_data = db['users'].get(session['user'], {})
    return await render_template('index.html', 
                               user=session['user'], 
                               nodes_limit=user_data.get('nodes', 0),
                               expires_at=user_data.get('expires_at'))

@app.route('/test')
async def test_page():
    return await render_template('test.html')

@app.route('/admin')
@login_required
async def admin():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    db = load_users()
    for name, user in db['users'].items():
        exp = datetime.datetime.fromisoformat(user['expires_at'])
        diff = exp - datetime.datetime.now()
        user['expires_at_human'] = "Expired" if diff.total_seconds() < 0 else f"{int(diff.total_seconds() // 3600)}h {int((diff.total_seconds() % 3600) // 60)}m"
            
    return await render_template('admin.html', users=db['users'], stats=get_stats(), proxies=load_proxies())

@app.route('/admin/proxy/add', methods=['POST'])
@login_required
async def admin_add_proxy():
    if session.get('role') != 'admin': return abort(403)
    form = await request.form
    proxy_str = form.get('proxy_str', '').strip()
    if not proxy_str: return redirect(url_for('admin'))
    
    parts = proxy_str.split(':')
    proxy_obj = {"server": f"http://{parts[0]}:{parts[1]}"}
    if len(parts) >= 4:
        proxy_obj["username"] = parts[2]
        proxy_obj["password"] = parts[3]
    
    proxies = load_proxies()
    proxies.append(proxy_obj)
    save_proxies(proxies)
    return redirect(url_for('admin'))

@app.route('/admin/proxy/delete', methods=['POST'])
@login_required
async def admin_delete_proxy():
    if session.get('role') != 'admin': return abort(403)
    form = await request.form
    index = int(form.get('index', -1))
    
    proxies = load_proxies()
    if 0 <= index < len(proxies):
        proxies.pop(index)
        save_proxies(proxies)
    return redirect(url_for('admin'))

@app.route('/admin/add', methods=['POST'])
@login_required
async def admin_add_user():
    if session.get('role') != 'admin': return abort(403)
    form = await request.form
    username = form.get('username')
    password = form.get('password')
    nodes = int(form.get('nodes', 10))
    minutes = int(form.get('minutes', 60))
    
    db = load_users()
    expires_at = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
    db['users'][username] = {
        "password": password,
        "nodes": nodes,
        "expires_at": expires_at.isoformat()
    }
    save_users(db)
    await broadcast_users()
    return redirect(url_for('admin'))

@app.route('/admin/update', methods=['POST'])
@login_required
async def admin_update_user():
    if session.get('role') != 'admin': return abort(403)
    form = await request.form
    username = form.get('username')
    nodes = int(form.get('nodes', 10))
    minutes = int(form.get('minutes', 0))
    
    db = load_users()
    if username in db['users']:
        db['users'][username]['nodes'] = nodes
        if minutes > 0:
            current_exp = datetime.datetime.fromisoformat(db['users'][username]['expires_at'])
            start_time = max(current_exp, datetime.datetime.now())
            db['users'][username]['expires_at'] = (start_time + datetime.timedelta(minutes=minutes)).isoformat()
        save_users(db)
        await sio.emit('config_update', {"nodes": nodes}, room=f"user_{username}")
        await broadcast_users()
    return redirect(url_for('admin'))

@app.route('/admin/delete', methods=['POST'])
@login_required
async def admin_delete_user():
    if session.get('role') != 'admin': return abort(403)
    form = await request.form
    username = form.get('username')
    db = load_users()
    if username in db['users']:
        del db['users'][username]
    save_users(db)
    await broadcast_users()
    await sio.emit('access_revoked', {}, room=f"user_{username}")
    return redirect(url_for('admin'))

@app.route('/admin/reset', methods=['POST'])
@login_required
async def admin_reset_all():
    if session.get('role') != 'admin': return abort(403)
    
    async with allocation_lock:
        node_leases.clear()
        for b_data in browsers.values():
            b_data['in_use_by'] = None
            b_data['assigned_at'] = None
        waiting_users.clear()
        
    await broadcast_to_all_users()
    await broadcast_stats()
    return redirect(url_for('admin'))

async def try_assign_nodes(username):
    """Attempt to assign free nodes to a user up to their limit via leases."""
    if not username or username == 'admin': return
    
    async with allocation_lock:
        db = load_users()
        if username not in db['users'] and username != 'admin':
            return

        # 1. Total limit for this user
        limit = db['users'][username].get('nodes', 0) if username != 'admin' else 999
        
        # 2. Check current leases for this user
        user_leased_node_ids = [nid for nid, uname in node_leases.items() if uname == username]
        
        if len(user_leased_node_ids) < limit:
            # 3. Find browsers that are NOT leased or are leased to me but not marked in_use_by
            # First, check if any registered browsers match my existing leases but are vacant
            for bid, bdata in browsers.items():
                nid = bdata['id']
                if node_leases.get(nid) == username and bdata.get('in_use_by') != username:
                    bdata['in_use_by'] = username
                    bdata['assigned_at'] = datetime.datetime.now().isoformat()
            
            # Recalculate
            user_leased_node_ids = [nid for nid, uname in node_leases.items() if uname == username]
            
            if len(user_leased_node_ids) < limit:
                # 4. Find completely free nodes (not in browsers and not in node_leases)
                # Or registered browsers that have no lease
                all_registered_nids = {b['id'] for b in browsers.values()}
                leased_nids = set(node_leases.keys())
                
                # Available registered nodes with no lease
                free_reg_nodes = [b for b in browsers.values() if b['id'] not in leased_nids]
                free_reg_nodes.sort(key=lambda x: x['id'])
                
                needed = limit - len(user_leased_node_ids)
                to_lease = free_reg_nodes[:needed]
                
                for node in to_lease:
                    node_leases[node['id']] = username
                    node['in_use_by'] = username
                    node['assigned_at'] = datetime.datetime.now().isoformat()
                    logger.info(f"Node {node['id']} LEASED and assigned to {username}")
                
                # 5. If still need more, add to waiting if not already there
                if len(user_leased_node_ids) + len(to_lease) < limit:
                    if username not in waiting_users and username != 'admin':
                        waiting_users.append(username)
                        logger.info(f"User {username} added to waiting list (needs {limit - (len(user_leased_node_ids) + len(to_lease))} more)")
                else:
                    if username in waiting_users: waiting_users.remove(username)
        else:
            if username in waiting_users: waiting_users.remove(username)

    await broadcast_stats()
    await broadcast_to_all_users()

async def broadcast_to_all_users():
    for user_sid, uname in list(sid_to_user.items()):
        if user_sid not in browsers:
             await emit_user_browsers(user_sid, uname)

async def reassign_freed_nodes():
    """Check waiting list and assign available nodes via leases."""
    async with allocation_lock:
        if not waiting_users: return
        
        # Nodes that are registered but have NO lease
        leased_nids = set(node_leases.keys())
        free_reg_nodes = [b for b in browsers.values() if b['id'] not in leased_nids]
        
        if not free_reg_nodes: return
        
        for username in list(waiting_users):
            if not free_reg_nodes: break
            
            db = load_users()
            limit = db['users'].get(username, {}).get('nodes', 0)
            current_leases = len([nid for nid, uname in node_leases.items() if uname == username])
            
            while current_leases < limit and free_reg_nodes:
                node = free_reg_nodes.pop(0)
                node_leases[node['id']] = username
                node['in_use_by'] = username
                node['assigned_at'] = datetime.datetime.now().isoformat()
                current_leases += 1
                logger.info(f"Node {node['id']} leased to waiting user {username}")
            
            if current_leases >= limit:
                waiting_users.remove(username)

    await broadcast_stats()
    await broadcast_to_all_users()

def get_user_allowed_nodes(username):
    """Get unique nodes this user is currently assigned to (via leases)"""
    if username == 'admin':
        return sorted(list(browsers.values()), key=lambda x: (x.get('id', ''), x.get('sid', '')))
    
    # Return registered browsers whose ID is leased to this user
    return [b for b in browsers.values() if node_leases.get(b['id']) == username]

async def emit_user_browsers(sid, username=None):
    if not username: return
    db = load_users()
    limit = db['users'].get(username, {}).get('nodes', 0) if username != 'admin' else 999
    allowed = get_user_allowed_nodes(username)
    
    # Enrich nodes with is_held status
    for node in allowed:
        node['is_held'] = node['id'] in held_nodes
    
    # Calculate how many nodes they are waiting for (leases vs limit)
    current_lease_count = len([nid for nid, uname in node_leases.items() if uname == username])
    waiting_count = max(0, limit - current_lease_count) if username != 'admin' else 0
    
    await sio.emit('browser_list_update', {
        'nodes': allowed,
        'waiting_count': waiting_count,
        'limit': limit
    }, room=sid)

@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")

@sio.on('set_user')
async def set_user(sid, data):
    username = data.get('username')
    role = data.get('role', 'user')
    sid_to_user[sid] = username
    user_last_seen[username] = datetime.datetime.now()  # Track last seen
    
    if role == 'admin' or username == 'admin':
        await sio.enter_room(sid, 'role_admin')
        await broadcast_stats()
    
    await sio.enter_room(sid, f"user_{username}")
    await try_assign_nodes(username)

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    
    # If a browser disconnected
    if sid in browsers:
        del browsers[sid]
    
    # If a user disconnected, free their nodes only if this was their last connection
    username = sid_to_user.get(sid)
    if username:
        del sid_to_user[sid]
        
        # Check if they have other active connections
        other_connections = [s for s, u in sid_to_user.items() if u == username]
        
        if not other_connections:
            if username in waiting_users:
                waiting_users.remove(username)
                
            async with allocation_lock:
                # 1. Clear in_use_by for registered browsers
                for b_sid, b_data in browsers.items():
                    if b_data.get('in_use_by') == username:
                        b_data['in_use_by'] = None
                        b_data['assigned_at'] = None
                
                # 2. Release LEASES ONLY for nodes that are NOT held
                if username != 'admin':
                    expired_leases = [nid for nid, uname in node_leases.items() if uname == username and nid not in held_nodes]
                    for nid in expired_leases:
                        del node_leases[nid]
                    logger.info(f"Freed {len(expired_leases)} nodes (user {username} disconnected, {len([n for n in held_nodes if held_nodes.get(n, {}).get('username') == username])} nodes held)")
            
            await reassign_freed_nodes()
        else:
            logger.info(f"User {username} disconnected SID {sid}, but still has {len(other_connections)} other active connections.")

    # Refresh all remaining users (use list() for safe iteration)
    for user_sid, uname in list(sid_to_user.items()):
        if user_sid not in browsers: # only if it's a UI client
            await emit_user_browsers(user_sid, uname)
    await broadcast_stats()

@sio.event
async def register_browser(sid, data):
    browser_id = data.get('id', sid)
    proxy_info = data.get('proxy_info', 'Direct')  # New: receive proxy info from node
    
    # Check if this browser ID is already leased to someone
    leased_to = node_leases.get(browser_id)
    
    browsers[sid] = {
        'id': browser_id, 
        'sid': sid, 
        'in_use_by': leased_to, 
        'assigned_at': datetime.datetime.now().isoformat() if leased_to else None,
        'proxy_info': proxy_info,
        'is_held': browser_id in held_nodes
    }
    logger.info(f"Browser registered: {browser_id} (Leased to: {leased_to}, Proxy: {proxy_info})")
    
    await reassign_freed_nodes()
    await broadcast_stats()
    await broadcast_to_all_users()

def is_authorized(sid, target_sids):
    """Check if the client (sid) is authorized to control target(s) based on leases"""
    username = sid_to_user.get(sid)
    if not username: 
        logger.warning(f"Unauthorized: SID {sid} has no username mapping")
        return False
    
    if username == 'admin': return True
    
    if isinstance(target_sids, str):
        targets = [target_sids]
    else:
        targets = target_sids

    for t_sid in targets:
        browser = browsers.get(t_sid)
        if not browser:
            logger.warning(f"Control blocked: Node {t_sid} not found")
            asyncio.create_task(sio.emit('control_status', {'type': 'error', 'msg': f"Node {t_sid} disconnected"}, room=sid))
            return False
            
        nid = browser['id']
        if node_leases.get(nid) != username:
            logger.warning(f"Blocked: User '{username}' attempted to control node {nid} leased to {node_leases.get(nid)}")
            asyncio.create_task(sio.emit('control_status', {'type': 'error', 'msg': f"Access Denied: Node {nid} is not yours"}, room=sid))
            return False

    return True

@sio.event
async def offer(sid, data):
    target = data.get('target')
    username = sid_to_user.get(sid)
    
    if not target or not username: return

    if is_authorized(sid, target):
        # Already assigned in try_assign_nodes/reassign_freed_nodes
        # We just trigger the restart and forward the offer
        await sio.emit('restart_browser', {}, room=target)
        logger.info(f"Triggered browser restart on {target} for {username}")
        await asyncio.sleep(0.5) 
        await sio.emit('offer', {'sdp': data['sdp'], 'type': data['type'], 'from': sid}, room=target)

@sio.event
async def answer(sid, data):
    target = data.get('target')
    if target: await sio.emit('answer', {'sdp': data['sdp'], 'type': data['type'], 'from': sid}, room=target)

@sio.event
async def ice_candidate(sid, data):
    target = data.get('target')
    if target and is_authorized(sid, target):
        await sio.emit('ice_candidate', {'candidate': data['candidate'], 'from': sid}, room=target)

@sio.event
async def control_event(sid, data):
    targets = data.get('target')
    if targets and is_authorized(sid, targets):
        if isinstance(targets, list):
            for t in targets:
                # Copy data and set target to specific for each node
                node_data = data.copy()
                node_data['target'] = t
                await sio.emit('control_event', node_data, room=t)
        else:
            await sio.emit('control_event', data, room=targets)

@sio.on('request_proxy')
async def request_proxy(sid, data):
    import random
    proxies = load_proxies()
    if not proxies:
        return None
    
    selected = random.choice(proxies)
    logger.info(f"Assigned proxy {selected['server']} to node {sid}")
    return selected

# --- Admin Management Tools ---

@sio.on('admin_restart_all')
async def admin_restart_all(sid, data):
    if sid_to_user.get(sid) != 'admin': return
    logger.info("Admin triggered global browser restart")
    await sio.emit('restart_browser', {}, room='all_nodes') # Note: browsers should join 'all_nodes'
    # Fallback to individual emits if room not joined
    for b_sid in list(browsers.keys()):
        await sio.emit('restart_browser', {}, room=b_sid)
    await asyncio.sleep(0.5)
    await broadcast_to_all_users()

@sio.on('admin_restart_node')
async def admin_restart_node(sid, data):
    if sid_to_user.get(sid) != 'admin': return
    target = data.get('target')
    if target in browsers:
        logger.info(f"Admin triggered restart for node {browsers[target]['id']}")
        await sio.emit('restart_browser', {}, room=target)

@sio.on('admin_kick_user')
async def admin_kick_user(sid, data):
    if sid_to_user.get(sid) != 'admin': return
    target_user = data.get('username')
    if not target_user or target_user == 'admin': return
    
    logger.info(f"Admin kicking user: {target_user}")
    
    async with allocation_lock:
        # Clear leases
        expired_leases = [nid for nid, uname in node_leases.items() if uname == target_user]
        for nid in expired_leases:
            del node_leases[nid]
        
        # Clear in_use_by
        for b_sid, b_data in browsers.items():
            if b_data.get('in_use_by') == target_user:
                b_data['in_use_by'] = None
                b_data['assigned_at'] = None
    
    await sio.emit('access_revoked', {}, room=f"user_{target_user}")
    await reassign_freed_nodes()
    await broadcast_to_all_users()

# --- Node Holding Feature ---

@sio.on('toggle_hold_node')
async def toggle_hold_node(sid, data):
    """Toggle hold status for a node. Only the user who owns the lease can hold it."""
    username = sid_to_user.get(sid)
    node_id = data.get('node_id')
    
    if not username or not node_id:
        return {'success': False, 'error': 'Invalid request'}
    
    # Verify ownership
    if node_leases.get(node_id) != username and username != 'admin':
        return {'success': False, 'error': 'You do not own this node'}
    
    async with allocation_lock:
        if node_id in held_nodes:
            # Unhold
            del held_nodes[node_id]
            logger.info(f"Node {node_id} UNHELD by {username}")
        else:
            # Hold
            held_nodes[node_id] = {
                'username': username,
                'held_at': datetime.datetime.now().isoformat(),
                'expiry': (datetime.datetime.now() + datetime.timedelta(minutes=HOLD_TIMEOUT_MINUTES)).isoformat()
            }
            logger.info(f"Node {node_id} HELD by {username} (expires in {HOLD_TIMEOUT_MINUTES}m)")
    
    # Update browser state
    for b_sid, b_data in browsers.items():
        if b_data['id'] == node_id:
            b_data['is_held'] = node_id in held_nodes
    
    await broadcast_to_all_users()
    return {'success': True, 'is_held': node_id in held_nodes}

@sio.on('change_proxy_request')
async def change_proxy_request(sid, data):
    """Request a node to restart with a different proxy or no proxy."""
    username = sid_to_user.get(sid)
    target_sid = data.get('target')
    new_proxy = data.get('proxy')  # None for direct, or a proxy object
    
    if not username or not target_sid:
        return {'success': False, 'error': 'Invalid request'}
    
    if not is_authorized(sid, target_sid):
        return {'success': False, 'error': 'Unauthorized'}
    
    # Send proxy change request to the node
    await sio.emit('set_proxy', {'proxy': new_proxy}, room=target_sid)
    logger.info(f"Proxy change requested for {target_sid} by {username}: {new_proxy}")
    return {'success': True}

@sio.on('refresh_page')
async def refresh_page(sid, data):
    """Request a node to refresh its current page."""
    username = sid_to_user.get(sid)
    target_sid = data.get('target')
    
    if not username or not target_sid:
        return
    
    if is_authorized(sid, target_sid):
        await sio.emit('control_event', {'type': 'reload'}, room=target_sid)
        logger.info(f"Page refresh requested for {target_sid} by {username}")

@sio.on('request_fallback')
async def request_fallback(sid, data):
    target = data.get('target')
    if is_authorized(sid, target):
        logger.info(f"Fallback requested for node {target}")
        await sio.emit('request_fallback', {}, room=target)

@sio.on('stop_fallback')
async def stop_fallback(sid, data):
    target = data.get('target')
    if is_authorized(sid, target):
        await sio.emit('stop_fallback', {}, room=target)

@sio.on('request_keyframe')
async def request_keyframe(sid, data):
    target = data.get('target')
    if is_authorized(sid, target):
        await sio.emit('request_keyframe', {}, room=target)

@sio.on('video_data')
async def relay_video(sid, data):
    """Relay binary video data from Node to User and Admins"""
    if sid in browsers:
        # Relay to admins for monitoring
        await sio.emit('video_frame', {'sid': sid, 'data': data}, room='role_admin')
        
        username = browsers[sid].get('in_use_by')
        if username and username != 'admin':
            await sio.emit('video_frame', {'sid': sid, 'data': data}, room=f"user_{username}")


# --- Background Cleanup Task ---

async def background_cleanup_task():
    """Background task to clean up expired holds and accounts."""
    while True:
        await asyncio.sleep(60)  # Run every minute
        try:
            now = datetime.datetime.now()
            
            # 1. Check for expired holds (user offline for too long)
            async with allocation_lock:
                expired_hold_ids = []
                for node_id, hold_info in list(held_nodes.items()):
                    username = hold_info.get('username')
                    
                    # Check if user is still connected
                    user_online = any(u == username for u in sid_to_user.values())
                    
                    if not user_online:
                        # Check expiry
                        expiry = datetime.datetime.fromisoformat(hold_info.get('expiry'))
                        if now > expiry:
                            expired_hold_ids.append(node_id)
                            logger.info(f"Hold expired for node {node_id} (user {username} offline too long)")
                
                # Release expired holds
                for node_id in expired_hold_ids:
                    del held_nodes[node_id]
                    if node_id in node_leases:
                        del node_leases[node_id]
                    # Clear in_use_by
                    for b_sid, b_data in browsers.items():
                        if b_data['id'] == node_id:
                            b_data['in_use_by'] = None
                            b_data['assigned_at'] = None
                            b_data['is_held'] = False
                
                if expired_hold_ids:
                    await reassign_freed_nodes()
            
            # 2. Check for expired user accounts
            db = load_users()
            for username, user_data in list(db['users'].items()):
                expires_at = datetime.datetime.fromisoformat(user_data['expires_at'])
                if now > expires_at:
                    # Kick the user
                    logger.info(f"Account expired for {username}, kicking...")
                    await sio.emit('access_revoked', {}, room=f"user_{username}")
                    
                    # Clear their leases and holds
                    async with allocation_lock:
                        expired_leases = [nid for nid, uname in node_leases.items() if uname == username]
                        for nid in expired_leases:
                            if nid in held_nodes:
                                del held_nodes[nid]
                            del node_leases[nid]
                        
                        for b_sid, b_data in browsers.items():
                            if b_data.get('in_use_by') == username:
                                b_data['in_use_by'] = None
                                b_data['assigned_at'] = None
                                b_data['is_held'] = False
                    
                    await reassign_freed_nodes()
            
            await broadcast_to_all_users()
            await broadcast_stats()
            
        except Exception as e:
            logger.error(f"Background cleanup error: {e}")

# Start background task when app starts
@app.before_serving
async def startup():
    asyncio.create_task(background_cleanup_task())
    logger.info("Background cleanup task started")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
