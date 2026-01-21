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
# Use pbkdf2_sha256 for better compatibility on cloud platforms
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
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
            is_admin INTEGER DEFAULT 0
        )
    """)
    # Create default admin if no users exist
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        admin_pass = "joemake"
        cursor.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                       ("admin", pwd_context.hash(admin_pass), 1))
    conn.commit()
    conn.close()

init_db()

# Session storage (In-memory for simplicity, or could be in DB)
# session_id: username
active_sessions: Dict[str, str] = {}

class UserCreate(BaseModel):
    username: str
    password: str
    is_admin: bool = False

class UserUpdate(BaseModel):
    password: Optional[str] = None
    is_admin: Optional[bool] = None

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
    browsers: List[Any]  # Can be IDs or complex objects
    status: str = "healthy"
    error: Optional[str] = None

# Real-time dashboard clients
hub_clients: List[WebSocket] = []

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

async def broadcast_hub():
    """Prune offline nodes and broadcast state"""
    now = time.time()
    pruned_nodes = {nid: data for nid, data in nodes.items() if now - data["last_seen"] < NODE_TIMEOUT}
    
    msg = {"type": "update", "nodes": pruned_nodes}
    for client in list(hub_clients):
        try: 
            await client.send_json(msg)
        except: 
            hub_clients.remove(client)

# ======================================================================
# API ENDPOINTS
# ======================================================================

@app.post("/register")
async def register_node(info: NodeInfo):
    """Nodes call this to announce presence"""
    old_data = nodes.get(info.node_id)
    new_browsers = info.browsers
    
    # Check if anything changed besides the heartbeat timestamp
    changed = False
    if not old_data:
        changed = True
    elif (old_data.get("browsers") != new_browsers or 
          old_data.get("url") != info.url or 
          old_data.get("status") != info.status or
          old_data.get("error") != info.error):
        changed = True
        
    nodes[info.node_id] = {
        "url": info.url,
        "browsers_count": info.browsers_count,
        "browsers": new_browsers,
        "status": info.status,
        "error": info.error,
        "last_seen": time.time()
    }
    
    if changed:
        print(f" [HUB] Node Registered: {info.node_id} at {info.url} ({info.browsers_count} browsers)")
        asyncio.create_task(broadcast_hub())
    return {"status": "ok"}

@app.websocket("/ws/hub")
async def hub_endpoint(websocket: WebSocket):
    await websocket.accept()
    hub_clients.append(websocket)
    # Send immediate state
    now = time.time()
    active = {nid: data for nid, data in nodes.items() if now - data["last_seen"] < NODE_TIMEOUT}
    await websocket.send_json({"type": "update", "nodes": active})
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in hub_clients: 
            hub_clients.remove(websocket)

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
    
    result = await send_node_command(node_id, "delete_profile", {"profile_id": profile_id})
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
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.post(f"{node_url}/refresh")
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/request_browser")
async def request_browser(count: int = 1, mode: str = "ephemeral", profile_id: str = None, current_user: sqlite3.Row = Depends(get_current_user)):
    """Allocates multiple browsers across the cluster"""
    if count < 1 or count > 100: 
        count = 1
    
    results = []
    for _ in range(count):
        now = time.time()
        active_node_ids = [nid for nid, data in nodes.items() if now - data["last_seen"] < NODE_TIMEOUT]
        
        if not active_node_ids:
            break
        
        # Smart load balancing - pick node with fewest browsers
        best_node_id = min(active_node_ids, key=lambda nid: nodes[nid]["browsers_count"])
        
        # Try WebSocket Command first (Works through firewall/cloud)
        cmd_result = await send_node_command(best_node_id, "create", {"mode": mode, "profile_id": profile_id})
        if cmd_result:
            results.append(cmd_result)
            continue

        # Legacy HTTP Fallback (Only works if server can see node IP)
        node_url = nodes[best_node_id]["url"]
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                payload = {"mode": mode, "profile_id": profile_id}
                resp = await client.post(f"{node_url}/create", json=payload)
                if resp.status_code == 200:
                    results.append(resp.json())
        except Exception as e: 
            print(f"Failed to create browser on {best_node_id}: {e}")
            continue
        
    return {"results": results, "count": len(results)}

@app.get("/api/nodes")
async def get_nodes(current_user: sqlite3.Row = Depends(get_current_user)):
    """Get current node status"""
    now = time.time()
    active_nodes = {nid: data for nid, data in nodes.items() if now - data["last_seen"] < NODE_TIMEOUT}
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
    return {
        "username": current_user["username"],
        "is_admin": bool(current_user["is_admin"])
    }

# ======================================================================
# USER MANAGEMENT ROUTES (ADMIN ONLY)
# ======================================================================

@app.get("/api/users")
async def list_users(admin: sqlite3.Row = Depends(get_admin_user)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    users = conn.execute("SELECT id, username, is_admin FROM users").fetchall()
    conn.close()
    return [dict(u) for u in users]

@app.post("/api/users")
async def create_user(user: UserCreate, admin: sqlite3.Row = Depends(get_admin_user)):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                     (user.username, pwd_context.hash(user.password), 1 if user.is_admin else 0))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        conn.close()
    return {"status": "ok"}

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int, admin: sqlite3.Row = Depends(get_admin_user)):
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    if not conn.fetchone:
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

