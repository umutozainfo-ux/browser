import secrets
import time
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import httpx
import uvicorn
import sqlite3
from passlib.context import CryptContext
from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect, Form, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse

# ======================================================================
# STEALTHNODE HUB - MODERN MULTI-PAGE ARCHITECTURE
# ======================================================================

DB_PATH = "hub_data.db"
# Use pbkdf2_sha256 for better compatibility on cloud platforms, keep bcrypt for legacy识别
pwd_context = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated="auto")
app = FastAPI(title="StealthNode Hub")

# Database initialization
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            browser_limit INTEGER DEFAULT 10
        )
    """)
    
    # Add browser_limit column if it doesn't exist (migration for existing DBs)
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN browser_limit INTEGER DEFAULT 10")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Create default admin if no users exist or if hash is unreadable
    admin_pass = "joemake"
    cursor.execute("SELECT password_hash FROM users WHERE username = 'admin'")
    row = cursor.fetchone()
    
    if not row:
        cursor.execute("INSERT INTO users (username, password_hash, is_admin, browser_limit) VALUES (?, ?, ?, ?)",
                       ("admin", pwd_context.hash(admin_pass), 1, -1))  # -1 = unlimited
    else:
        # Self-heal: If the existing hash format is unknown (e.g. after switching algorithms), reset it
        try:
            pwd_context.identify(row[0])
        except:
            cursor.execute("UPDATE users SET password_hash = ? WHERE username = 'admin'",
                           (pwd_context.hash(admin_pass),))
    
    conn.commit()
    conn.close()

@app.on_event("startup")
def on_startup():
    init_db()

# Session storage (In-memory for simplicity, or could be in DB)
# session_id: username
active_sessions: Dict[str, str] = {}

class UserCreate(BaseModel):
    username: str
    password: str
    is_admin: bool = False
    browser_limit: int = 10  # Default limit of 10 browsers per user

class UserUpdate(BaseModel):
    password: Optional[str] = None
    is_admin: Optional[bool] = None
    browser_limit: Optional[int] = None

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Get the directory where server.py is located
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Registry of all connected nodes
# { node_id: { url, browsers_count, browsers, last_seen } }
nodes: Dict[str, Dict[str, Any]] = {}
NODE_TIMEOUT = 15  # Seconds before a node is considered offline

class NodeInfo(BaseModel):
    node_id: str
    url: str
    browsers_count: int
    browsers: Optional[List[Any]] = None  # Delta: Only send when list changes
    status: str = "healthy"
    error: Optional[str] = None

# Real-time dashboard clients: { websocket: user_dict }
hub_clients: Dict[WebSocket, Dict[str, Any]] = {}

# Node Command System
# { node_id: WebSocket }
node_sockets: Dict[str, WebSocket] = {}
# { task_id: asyncio.Future }
pending_tasks: Dict[str, asyncio.Future] = {}

async def get_current_user(session_id: Optional[str] = Cookie(None)):
    if not session_id or session_id not in active_sessions:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")
    
    username = active_sessions[session_id]
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user

async def get_admin_user(current_user: sqlite3.Row = Depends(get_current_user)):
    if not current_user["is_admin"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user

# Scaling Control
last_broadcast_time = 0
BROADCAST_THROTTLE = 2.0 # Broadcast at most every 2 seconds to save CPU

async def broadcast_hub():
    """Prune offline nodes and broadcast SLIM summary (Scales to 100k+)"""
    global last_broadcast_time
    now = time.time()
    
    # Prune offline nodes from the global dict
    to_prune = [nid for nid, data in nodes.items() if now - data["last_seen"] > NODE_TIMEOUT]
    for nid in to_prune:
        nodes.pop(nid, None)
    
    if now - last_broadcast_time < BROADCAST_THROTTLE:
        return
    last_broadcast_time = now

    if not hub_clients:
        return

    for client, user in list(hub_clients.items()):
        # Filter nodes/browsers for this specific user
        filtered_nodes = {}
        total_user_browsers = 0
        
        for nid, data in nodes.items():
            if now - data["last_seen"] > NODE_TIMEOUT:
                continue
            
            # Strict Isolation: Only show and allow control of browsers owned by this user
            raw_browsers = data.get("browsers", [])
            filtered_browsers = []
            
            for b in raw_browsers:
                if isinstance(b, dict) and b.get("owner") == user["username"]:
                    # Ensure each browser has all required fields
                    filtered_browsers.append({
                        "id": b.get("id"),
                        "mode": b.get("mode", "ephemeral"),
                        "profile_id": b.get("profile_id"),
                        "owner": b.get("owner")
                    })
            
            user_browser_count = len(filtered_browsers)
            total_user_browsers += user_browser_count
            
            filtered_nodes[nid] = {
                "url": data.get("url"),  # Include URL for WebSocket connections
                "browsers_count": user_browser_count,
                "browsers": filtered_browsers,
                "status": data.get("status", "unknown"),
                "last_seen": data["last_seen"]
            }

        summary = {
            "type": "update",
            "nodes": filtered_nodes,
            "total_browsers": total_user_browsers,
            "total_nodes": len([n for n in filtered_nodes.values() if n["browsers_count"] > 0])
        }
        
        try:
            await client.send_json(summary)
        except:
            hub_clients.pop(client, None)

# ======================================================================
# API ENDPOINTS
# ======================================================================

@app.post("/register")
async def register_node(info: NodeInfo):
    """Nodes call this to announce presence. Optimized for high scale.
    
    This endpoint handles browser list synchronization carefully to ensure
    accurate counting:
    1. When node sends a full browser list, it becomes the source of truth
    2. When node sends delta (no browsers), we preserve existing list
    3. Browser IDs are deduplicated to prevent double-counting
    """
    old_data = nodes.get(info.node_id)
    
    # Delta Logic: If node didn't send browsers, use the old ones
    new_browsers = info.browsers
    if new_browsers is None and old_data:
        new_browsers = old_data.get("browsers", [])
    elif new_browsers is not None:
        # Node sent a full browser list - this is the authoritative source
        # Deduplicate by browser ID to ensure accurate counting
        seen_ids = set()
        deduplicated = []
        for browser in new_browsers:
            bid = browser.get("id") if isinstance(browser, dict) else browser
            if bid and bid not in seen_ids:
                seen_ids.add(bid)
                deduplicated.append(browser)
        new_browsers = deduplicated
    
    # Check if a broadcast is actually needed
    changed = False
    if not old_data:
        changed = True
    else:
        # Compare browser lists by ID to detect actual changes
        old_ids = {
            b.get("id") if isinstance(b, dict) else b
            for b in old_data.get("browsers", [])
        }
        new_ids = {
            b.get("id") if isinstance(b, dict) else b
            for b in (new_browsers or [])
        }
        if (old_ids != new_ids or
            info.status != old_data.get("status") or
            info.browsers_count != len(new_browsers or [])):
            changed = True
        
    nodes[info.node_id] = {
        "url": info.url,
        "browsers_count": len(new_browsers or []),  # Use actual count, not reported count
        "browsers": new_browsers or [],
        "status": info.status,
        "error": info.error,
        "last_seen": time.time()
    }
    
    if changed:
        asyncio.create_task(broadcast_hub())
    return {"status": "ok"}

@app.websocket("/ws/hub")
async def hub_endpoint(websocket: WebSocket, session_id: Optional[str] = Cookie(None)):
    if not session_id or session_id not in active_sessions:
        await websocket.close()
        return
        
    username = active_sessions[session_id]
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user_row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    
    if not user_row:
        await websocket.close()
        return

    user_info = {"username": username, "is_admin": bool(user_row["is_admin"])}
    await websocket.accept()
    hub_clients[websocket] = user_info
    
    # Send immediate filtered state
    asyncio.create_task(broadcast_hub())
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub_clients.pop(websocket, None)

@app.get("/api/profiles")
async def get_all_profiles(current_user: sqlite3.Row = Depends(get_current_user)):
    """Fetch profiles from all active nodes"""
    all_profiles = {}
    now = time.time()
    active_node_ids = [nid for nid, data in nodes.items() if now - data["last_seen"] < NODE_TIMEOUT]
    
    # Try WebSocket first for each node
    for nid in active_node_ids:
        res = await send_node_command(nid, "get_profiles")
        if res and "profiles" in res:
            all_profiles[nid] = res["profiles"]
            continue
            
        # Fallback to legacy HTTP
        try:
            node_url = nodes[nid]["url"]
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{node_url}/profiles")
                if resp.status_code == 200:
                    all_profiles[nid] = resp.json().get("profiles", [])
        except:
            pass
    
    return all_profiles

@app.websocket("/ws/node/{node_id}")
async def node_command_endpoint(websocket: WebSocket, node_id: str):
    """WebSocket used by nodes to receive commands from the hub"""
    await websocket.accept()
    node_sockets[node_id] = websocket
    print(f" [HUB] Command Channel Connected: {node_id}")
    try:
        while True:
            # Nodes send back task results
            data = await websocket.receive_json()
            task_id = data.get("task_id")
            if task_id in pending_tasks:
                future = pending_tasks.pop(task_id)
                if not future.done():
                    future.set_result(data.get("result"))
    except WebSocketDisconnect:
        print(f" [HUB] Command Channel Disconnected: {node_id}")
    finally:
        node_sockets.pop(node_id, None)

async def send_node_command(node_id: str, command: str, data: dict = None):
    """Sends a command to a node via its WebSocket and waits for response"""
    if node_id not in node_sockets:
        # Fallback to HTTP for local setups if WS isn't ready
        return None 
    
    task_id = f"task_{secrets.token_hex(4)}"
    msg = {"task_id": task_id, "command": command, "data": data or {}}
    
    future = asyncio.get_event_loop().create_future()
    pending_tasks[task_id] = future
    
    try:
        await node_sockets[node_id].send_json(msg)
        # Wait up to 30 seconds for the node to complete the task
        return await asyncio.wait_for(future, timeout=30.0)
    except Exception as e:
        print(f" [HUB] Task {task_id} failed: {e}")
        pending_tasks.pop(task_id, None)
        return None

@app.delete("/api/delete_profile/{node_id}/{profile_id:path}")
async def delete_remote_profile(node_id: str, profile_id: str, current_user: sqlite3.Row = Depends(get_current_user)):
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # We should ideally check if the user owns this profile or is an admin
    # For now, we trust the send_node_command will handle it or we check node data
    
    result = await send_node_command(node_id, "delete_profile", {
        "profile_id": profile_id,
        "owner": current_user["username"],
        "is_admin": bool(current_user["is_admin"])
    })
    if result: return result
    
    # Fallback to legacy HTTP
    node_url = nodes[node_id]["url"]
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.delete(f"{node_url}/profiles/{profile_id}")
        return resp.json()

@app.post("/api/refresh_node/{node_id}")
async def refresh_node(node_id: str, current_user: sqlite3.Row = Depends(get_current_user)):
    """Proxies a refresh request to a specific node"""
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    result = await send_node_command(node_id, "refresh")
    if result: return result

    node_url = nodes[node_id]["url"]
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{node_url}/refresh")
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/close_browser/{node_id}/{browser_id}")
async def close_node_browser(node_id: str, browser_id: str, current_user: sqlite3.Row = Depends(get_current_user)):
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Check ownership unless admin
    node_data = nodes[node_id]
    browsers_list = node_data.get("browsers", [])
    
    # Find the browser in the node's list
    browser = None
    browser_index = -1
    for i, b in enumerate(browsers_list):
        if isinstance(b, dict) and b.get("id") == browser_id:
            browser = b
            browser_index = i
            break
        elif b == browser_id:
            # Legacy case: fallback to old list format if needed
            browser = {"id": b, "owner": "system"}
            browser_index = i
            break

    if not browser:
        raise HTTPException(status_code=404, detail="Browser not found on this node")
    
    # Check ownership - admins can close any browser
    is_admin = bool(current_user["is_admin"])
    if not is_admin and browser.get("owner") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Unauthorized: You do not own this browser")

    result = await send_node_command(node_id, "close_browser", {"browser_id": browser_id})
    
    # Immediately update local state for accurate counting
    # This ensures the user sees the correct count right away
    if browser_index >= 0 and node_id in nodes:
        try:
            nodes[node_id]["browsers"].pop(browser_index)
            nodes[node_id]["browsers_count"] = len(nodes[node_id]["browsers"])
            # Trigger immediate broadcast so UI updates
            asyncio.create_task(broadcast_hub())
        except (IndexError, KeyError):
            pass  # Browser may have already been removed by heartbeat
    
    if result:
        return result
    
    # Fallback to direct HTTP
    node_url = nodes[node_id]["url"]
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(f"{node_url}/close/{browser_id}")
        return resp.json()

# ======================================================================
# BROWSER REQUEST SYSTEM - OPTIMIZED FOR SCALE
# ======================================================================

BATCH_SIZE = 50  # Process browsers in batches of 50 for optimal speed

async def create_browser_on_node(node_id: str, mode: str, profile_id: str, owner: str):
    """Helper to create a single browser on a node - returns result or None"""
    # Try WebSocket Command first (Works through firewall/cloud)
    cmd_result = await send_node_command(node_id, "create", {
        "mode": mode, 
        "profile_id": profile_id,
        "owner": owner
    })
    if cmd_result:
        return cmd_result

    # Legacy HTTP Fallback (Only works if server can see node IP)
    if node_id in nodes:
        node_url = nodes[node_id]["url"]
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                payload = {"mode": mode, "profile_id": profile_id, "owner": owner}
                resp = await client.post(f"{node_url}/create", json=payload)
                if resp.status_code == 200:
                    return resp.json()
        except Exception as e: 
            print(f"Failed to create browser on {node_id}: {e}")
    return None

def get_user_browser_count(username: str) -> int:
    """Count browsers currently owned by a user across all ACTIVE nodes only.
    
    This ensures accurate counting by:
    1. Only counting browsers from nodes that are online (within NODE_TIMEOUT)
    2. Properly checking browser ownership
    """
    count = 0
    now = time.time()
    for node_data in nodes.values():
        # Skip offline/stale nodes - their browser data may be outdated
        if now - node_data.get("last_seen", 0) > NODE_TIMEOUT:
            continue
        for browser in node_data.get("browsers", []):
            if isinstance(browser, dict) and browser.get("owner") == username:
                count += 1
    return count

@app.get("/api/request_browser")
async def request_browser(count: int = 1, mode: str = "ephemeral", profile_id: str = None, current_user: sqlite3.Row = Depends(get_current_user)):
    """
    Allocates multiple browsers across the cluster with optimized parallel processing.
    Supports massive scale (100k+) through batched concurrent requests.
    Enforces per-user browser limits.
    """
    username = current_user["username"]
    browser_limit = current_user["browser_limit"] if "browser_limit" in current_user.keys() else 10
    is_admin = bool(current_user["is_admin"])
    
    # Enforce browser limit (admins with -1 have unlimited)
    if browser_limit != -1 and not is_admin:
        current_count = get_user_browser_count(username)
        available = max(0, browser_limit - current_count)
        if available == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Browser limit reached. You have {current_count}/{browser_limit} browsers. Close some to request more."
            )
        count = min(count, available)
    
    # Validate count (allow larger batches for scaled operations)
    if count < 1:
        count = 1
    elif count > 100000:
        count = 100000  # Hard cap at 100k per request
    
    now = time.time()
    active_node_ids = [nid for nid, data in nodes.items() if now - data["last_seen"] < NODE_TIMEOUT]
    
    if not active_node_ids:
        raise HTTPException(status_code=503, detail="No active nodes available")
    
    results = []
    failed = 0
    
    # Process in batches for optimal performance
    for batch_start in range(0, count, BATCH_SIZE):
        batch_count = min(BATCH_SIZE, count - batch_start)
        
        # Create tasks for concurrent execution
        tasks = []
        for i in range(batch_count):
            # Round-robin with load balancing
            sorted_nodes = sorted(active_node_ids, key=lambda nid: nodes[nid]["browsers_count"] + len([t for t in tasks if t[1] == nid]))
            best_node = sorted_nodes[0]
            
            task = asyncio.create_task(
                create_browser_on_node(best_node, mode, profile_id, username)
            )
            tasks.append((task, best_node))
        
        # Wait for all tasks in this batch to complete
        batch_results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)
        
        for (task, node_id), result in zip(tasks, batch_results):
            if isinstance(result, Exception):
                failed += 1
            elif isinstance(result, dict) and "id" in result:
                results.append(result)
                # Optimistic Real-time Update - but check for duplicates first
                if node_id in nodes:
                    if "browsers" not in nodes[node_id]:
                        nodes[node_id]["browsers"] = []
                    
                    # Prevent duplicate entries by checking if browser ID already exists
                    existing_ids = {
                        b.get("id") if isinstance(b, dict) else b
                        for b in nodes[node_id]["browsers"]
                    }
                    if result.get("id") not in existing_ids:
                        # Add owner info to the result for proper filtering
                        browser_entry = {
                            "id": result.get("id"),
                            "mode": result.get("mode", "ephemeral"),
                            "profile_id": result.get("profile_id"),
                            "owner": username  # Use the requesting user's username
                        }
                        nodes[node_id]["browsers"].append(browser_entry)
                        nodes[node_id]["browsers_count"] = len(nodes[node_id]["browsers"])
            else:
                failed += 1
    
    # Trigger immediate broadcast so browsers appear instantly in UI
    asyncio.create_task(broadcast_hub())
    
    return {
        "results": results, 
        "count": len(results),
        "requested": count,
        "failed": failed,
        "remaining_limit": browser_limit - get_user_browser_count(username) if browser_limit != -1 else -1
    }

@app.get("/api/nodes")
async def get_nodes(current_user: sqlite3.Row = Depends(get_current_user)):
    """Get current node status filtered for the user"""
    now = time.time()
    active_nodes = {}
    for nid, data in nodes.items():
        if now - data["last_seen"] > NODE_TIMEOUT:
            continue

        # Filter browsers for this user
        raw_browsers = data.get("browsers", [])
        filtered_browsers = [b for b in raw_browsers if isinstance(b, dict) and b.get("owner") == current_user["username"]]

        active_nodes[nid] = {
            "url": data["url"],
            "browsers_count": len(filtered_browsers),
            "browsers": filtered_browsers,
            "status": data["status"],
            "last_seen": data["last_seen"]
        }

    return {"nodes": active_nodes, "count": len(active_nodes)}

# ======================================================================
# AUTHENTICATION ROUTES
# ======================================================================

@app.post("/auth/login")
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    
    if not user or not pwd_context.verify(password, user["password_hash"]):
        return HTMLResponse(content="""
            <script>
                alert('Invalid username or password');
                window.location.href = '/login';
            </script>
        """, status_code=401)
    
    session_id = secrets.token_hex(16)
    active_sessions[session_id] = username
    
    response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return response

@app.get("/auth/logout")
async def logout(response: Response, session_id: Optional[str] = Cookie(None)):
    if session_id in active_sessions:
        del active_sessions[session_id]
    response = RedirectResponse(url="/login")
    response.delete_cookie("session_id")
    return response

@app.get("/auth/me")
async def get_me(current_user: sqlite3.Row = Depends(get_current_user)):
    browser_limit = current_user["browser_limit"] if "browser_limit" in current_user.keys() else 10
    current_count = get_user_browser_count(current_user["username"])
    return {
        "username": current_user["username"],
        "is_admin": bool(current_user["is_admin"]),
        "browser_limit": browser_limit,
        "browsers_active": current_count,
        "browsers_available": browser_limit - current_count if browser_limit != -1 else -1
    }

@app.get("/api/my_browsers")
async def get_my_browsers(current_user: sqlite3.Row = Depends(get_current_user)):
    """Get detailed list of all browsers owned by the current user.
    
    This provides an accurate, authoritative count of browsers for the user,
    only counting browsers from active nodes.
    """
    username = current_user["username"]
    now = time.time()
    my_browsers = []
    
    for nid, data in nodes.items():
        # Skip offline/stale nodes
        if now - data.get("last_seen", 0) > NODE_TIMEOUT:
            continue
        
        for browser in data.get("browsers", []):
            if isinstance(browser, dict) and browser.get("owner") == username:
                my_browsers.append({
                    "id": browser.get("id"),
                    "node_id": nid,
                    "mode": browser.get("mode", "ephemeral"),
                    "profile_id": browser.get("profile_id"),
                    "owner": username
                })
    
    browser_limit = current_user["browser_limit"] if "browser_limit" in current_user.keys() else 10
    
    return {
        "browsers": my_browsers,
        "count": len(my_browsers),
        "limit": browser_limit,
        "available": browser_limit - len(my_browsers) if browser_limit != -1 else -1
    }

# ======================================================================
# USER MANAGEMENT ROUTES (ADMIN ONLY)
# ======================================================================

@app.get("/api/users")
async def list_users(admin: sqlite3.Row = Depends(get_admin_user)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    users = conn.execute("SELECT id, username, is_admin, browser_limit FROM users").fetchall()
    conn.close()
    
    # Add current browser count for each user
    result = []
    for u in users:
        user_dict = dict(u)
        user_dict["browsers_active"] = get_user_browser_count(u["username"])
        result.append(user_dict)
    return result

@app.post("/api/users")
async def create_user(user: UserCreate, admin: sqlite3.Row = Depends(get_admin_user)):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin, browser_limit) VALUES (?, ?, ?, ?)",
            (user.username, pwd_context.hash(user.password), 1 if user.is_admin else 0, user.browser_limit)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        conn.close()
    return {"status": "ok"}

@app.put("/api/users/{user_id}")
async def update_user(user_id: int, user: UserUpdate, admin: sqlite3.Row = Depends(get_admin_user)):
    """Update a user's properties (password, admin status, browser limit)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    existing = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
    
    updates = []
    params = []
    
    if user.password is not None:
        updates.append("password_hash = ?")
        params.append(pwd_context.hash(user.password))
    
    if user.is_admin is not None:
        updates.append("is_admin = ?")
        params.append(1 if user.is_admin else 0)
    
    if user.browser_limit is not None:
        updates.append("browser_limit = ?")
        params.append(user.browser_limit)
    
    if updates:
        params.append(user_id)
        conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    
    conn.close()
    return {"status": "ok"}

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int, admin: sqlite3.Row = Depends(get_admin_user)):
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    existing = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
    if not existing:
         conn.close()
         raise HTTPException(status_code=404, detail="User not found")
         
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/login", response_class=HTMLResponse)
async def login_page(session_id: Optional[str] = Cookie(None)):
    if session_id in active_sessions:
        return RedirectResponse(url="/")
    html_path = STATIC_DIR / "pages" / "login.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(content="Login page missing")

@app.get("/admin/users", response_class=HTMLResponse)
async def users_management_page(admin: sqlite3.Row = Depends(get_admin_user)):
    html_path = STATIC_DIR / "pages" / "users.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(content="User management page missing")

# ======================================================================
# PAGE ROUTES - MULTI-PAGE ARCHITECTURE
# ======================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(session_id: Optional[str] = Cookie(None)):
    if session_id not in active_sessions:
        return RedirectResponse(url="/login")
    """Command Center - Main browser control page"""
    html_path = STATIC_DIR / "pages" / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse(content=get_fallback_html("Command Center"), status_code=200)

@app.get("/vault", response_class=HTMLResponse)
async def vault_page(session_id: Optional[str] = Cookie(None)):
    if session_id not in active_sessions:
        return RedirectResponse(url="/login")
    """Profile Vault - Manage persistent profiles"""
    html_path = STATIC_DIR / "pages" / "vault.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse(content=get_fallback_html("Profile Vault"), status_code=200)

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(session_id: Optional[str] = Cookie(None)):
    if session_id not in active_sessions:
        return RedirectResponse(url="/login")
    """Settings - System configuration and info"""
    html_path = STATIC_DIR / "pages" / "settings.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse(content=get_fallback_html("Settings"), status_code=200)

def get_fallback_html(page_name: str) -> str:
    """Fallback HTML if static files are not found"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>StealthNode | {page_name}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #030712;
                color: #f1f5f9;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin: 0;
                padding: 20px;
            }}
            .error-box {{
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 12px;
                padding: 32px;
                max-width: 500px;
                text-align: center;
            }}
            h1 {{
                color: #ef4444;
                margin-bottom: 16px;
            }}
            p {{
                color: #94a3b8;
                line-height: 1.6;
            }}
            code {{
                background: rgba(0,0,0,0.3);
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="error-box">
            <h1>Static Files Not Found</h1>
            <p>
                The UI static files were not found. Make sure you have the 
                <code>static/</code> directory with all CSS, JS, and HTML files.
            </p>
            <p>
                Expected path: <code>{STATIC_DIR}</code>
            </p>
        </div>
    </body>
    </html>
    """

# ======================================================================
# HEALTH CHECK
# ======================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    now = time.time()
    active_nodes = sum(1 for data in nodes.values() if now - data["last_seen"] < NODE_TIMEOUT)
    return {
        "status": "healthy",
        "active_nodes": active_nodes,
        "total_browsers": sum(len(n.get("browsers", [])) for n in nodes.values())
    }

if __name__ == "__main__":
    print("""
    ============================================================
    
         STEALTHNODE COMMAND CENTER v2.0
         Modern Multi-Page UI
    
    ============================================================
    """)
    print(f"    Starting server at http://0.0.0.0:7860")
    print(f"    Static files: {STATIC_DIR}")
    print("""
    Pages:
      - Command Center: http://localhost:7860/
      - Profile Vault:  http://localhost:7860/vault
      - Settings:       http://localhost:7860/settings
      - Health Check:   http://localhost:7860/health
    """)
    uvicorn.run(app, host="0.0.0.0", port=7860)


