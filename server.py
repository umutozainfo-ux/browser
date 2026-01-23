import secrets
import time
import asyncio
import os
import logging
import functools
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from enum import Enum, auto
from dataclasses import dataclass, field
from pydantic import BaseModel
import httpx
import uvicorn
import sqlite3
from passlib.context import CryptContext
from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect, Form, Response, Cookie, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
try:
    from redis import asyncio as aioredis
except Exception:
    aioredis = None

# ======================================================================
# STEALTHNODE HUB - MODERN MULTI-PAGE ARCHITECTURE
# Professional Connection Management System
# ======================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger("StealthHub")

# ======================================================================
# CONNECTION STATE MANAGEMENT
# ======================================================================
class ConnectionState(Enum):
    """Represents the state of a connection"""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()

@dataclass
class NodeConnection:
    """Tracks the state and metadata of a node connection"""
    node_id: str
    url: str
    state: ConnectionState = ConnectionState.DISCONNECTED
    websocket: Optional[WebSocket] = None
    last_seen: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    browsers: List[Dict[str, Any]] = field(default_factory=list)
    browsers_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    reconnect_attempts: int = 0
    status: str = "unknown"
    health: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "node_id": self.node_id,
            "url": self.url,
            "state": self.state.name,
            "last_seen": self.last_seen,
            "browsers_count": self.browsers_count,
            "browsers": self.browsers,
            "status": self.status,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "health": self.health
        }

@dataclass
class ClientConnection:
    """Tracks the state of a UI client connection"""
    websocket: WebSocket
    user: Dict[str, Any]
    state: ConnectionState = ConnectionState.CONNECTED
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    messages_sent: int = 0
    messages_received: int = 0

class ConnectionManager:
    """
    Professional connection manager for handling all WebSocket connections.
    Implements proper state tracking, reconnection logic, and health monitoring.
    """
    
    def __init__(self):
        self._nodes: Dict[str, NodeConnection] = {}
        self._clients: Dict[WebSocket, ClientConnection] = {}
        self._node_sockets: Dict[str, WebSocket] = {}
        self._pending_tasks: Dict[str, asyncio.Future] = {}
        # map task_id -> node_id for cleanup when a node disconnects
        self._pending_task_nodes: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._broadcast_throttle = 2.0
        self._last_broadcast = 0
        self._node_timeout = 30  # seconds (increased for stability during heavy spawn)
        
        logger.info("ConnectionManager initialized")
    
    # ==================== Node Management ====================
    
    async def register_node(self, node_id: str, url: str, browsers: List[Dict], 
                           browsers_count: int, status: str = "healthy", 
                           error: Optional[str] = None, health: Optional[Dict] = None) -> NodeConnection:
        """Register or update a node connection"""
        async with self._lock:
            changed = False
            
            if node_id in self._nodes:
                node = self._nodes[node_id]
                old_browser_ids = {b.get("id") for b in node.browsers if isinstance(b, dict)}
                
                # Update existing node
                node.url = url
                node.last_seen = time.time()
                node.last_heartbeat = time.time()
                node.status = status
                node.health = health
                
                if error:
                    node.error_count += 1
                    node.last_error = error
                else:
                    node.error_count = 0
                    node.last_error = None
                
                # Handle browser list updates
                if browsers is not None:
                    # Deduplicate browsers
                    seen_ids = set()
                    deduplicated = []
                    for browser in browsers:
                        bid = browser.get("id") if isinstance(browser, dict) else browser
                        if bid and bid not in seen_ids:
                            seen_ids.add(bid)
                            deduplicated.append(browser)
                    node.browsers = deduplicated
                    node.browsers_count = len(deduplicated)
                    
                    # Check if browser list changed
                    new_browser_ids = {b.get("id") for b in deduplicated if isinstance(b, dict)}
                    if old_browser_ids != new_browser_ids:
                        logger.info(f"Node {node_id} browser list changed: {len(old_browser_ids)} -> {len(new_browser_ids)}")
                        changed = True
                
                node.state = ConnectionState.CONNECTED
            else:
                # Create new node connection
                # If this node reports browsers that belong to an existing (possibly stale)
                # node, migrate those browser entries and preserve ownership. This handles
                # the case where a node restarts and obtains a new node_id but continues
                # to host the same browser instances.
                incoming_ids = {b.get("id") for b in (browsers or []) if isinstance(b, dict)}
                best_match_id = None
                best_overlap = 0

                for nid, n in self._nodes.items():
                    if not n.browsers:
                        continue
                    existing_ids = {b.get("id") for b in n.browsers if isinstance(b, dict)}
                    overlap = len(incoming_ids & existing_ids)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match_id = nid

                # If we found a node with overlapping browser ids, migrate owner info
                if best_match_id and best_overlap > 0:
                    matched = self._nodes[best_match_id]
                    logger.info(f"Detected node migration: incoming {node_id} overlaps {best_overlap} browsers with {best_match_id}; migrating ownerships")

                    # Ensure incoming browser objects include owner/profile info from matched node
                    incoming_by_id = {}
                    for b in (browsers or []):
                        if isinstance(b, dict) and b.get("id"):
                            incoming_by_id[b.get("id")] = b

                    for b in list(matched.browsers):
                        if not isinstance(b, dict):
                            continue
                        bid = b.get("id")
                        if bid in incoming_by_id:
                            inc = incoming_by_id[bid]
                            # Preserve owner/profile if missing on incoming
                            if not inc.get("owner") and b.get("owner"):
                                inc["owner"] = b.get("owner")
                            if not inc.get("profile_id") and b.get("profile_id"):
                                inc["profile_id"] = b.get("profile_id")
                            # Remove migrated browser from the old node to avoid duplicates
                            try:
                                matched.browsers.remove(b)
                            except ValueError:
                                pass

                    matched.browsers_count = len(matched.browsers)

                node = NodeConnection(
                    node_id=node_id,
                    url=url,
                    state=ConnectionState.CONNECTED,
                    last_seen=time.time(),
                    browsers=browsers or [],
                    browsers_count=browsers_count,
                    status=status,
                    health=health
                )
                self._nodes[node_id] = node
                logger.info(f"New node registered: {node_id} at {url}")
                changed = True
            
            return node, changed
    
    async def connect_node_websocket(self, node_id: str, websocket: WebSocket) -> bool:
        """Connect a node's command WebSocket"""
        async with self._lock:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                node.websocket = websocket
                node.state = ConnectionState.CONNECTED
                node.last_seen = time.time()
                node.last_heartbeat = time.time()
                # If node already has browsers, trigger an immediate broadcast so
                # UI clients see restored browsers without delay.
                try:
                    asyncio.create_task(self.broadcast_to_clients(force=True))
                except Exception:
                    pass

            self._node_sockets[node_id] = websocket
            logger.info(f"Node {node_id} command channel connected")
            return True
    
    async def disconnect_node_websocket(self, node_id: str):
        """Disconnect a node's command WebSocket"""
        async with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].websocket = None
                self._nodes[node_id].state = ConnectionState.RECONNECTING
            
            self._node_sockets.pop(node_id, None)

            # Cancel or fail pending tasks that were targeted at this node
            to_cancel = [tid for tid, nid in list(self._pending_task_nodes.items()) if nid == node_id]
            for tid in to_cancel:
                future = self._pending_tasks.pop(tid, None)
                self._pending_task_nodes.pop(tid, None)
                if future and not future.done():
                    try:
                        future.set_exception(RuntimeError("Node disconnected"))
                    except Exception:
                        pass

            logger.info(f"Node {node_id} command channel disconnected and cleaned {len(to_cancel)} pending tasks")
    
    def get_node(self, node_id: str) -> Optional[NodeConnection]:
        """Get a node connection by ID"""
        return self._nodes.get(node_id)
    
    def get_active_nodes(self) -> Dict[str, NodeConnection]:
        """Get all active nodes (within timeout)"""
        now = time.time()
        return {
            nid: node for nid, node in self._nodes.items()
            if now - node.last_seen < self._node_timeout
        }
    
    async def prune_stale_nodes(self) -> List[str]:
        """Remove nodes that haven't been seen recently"""
        async with self._lock:
            now = time.time()
            stale = [
                nid for nid, node in self._nodes.items()
                if now - node.last_seen > self._node_timeout
            ]
            for nid in stale:
                self._nodes[nid].state = ConnectionState.DISCONNECTED
                logger.warning(f"Node {nid} marked as stale (last seen {now - self._nodes[nid].last_seen:.1f}s ago)")
            return stale
    
    # ==================== Client Management ====================
    
    async def connect_client(self, websocket: WebSocket, user: Dict[str, Any]) -> ClientConnection:
        """Register a new UI client connection"""
        async with self._lock:
            client = ClientConnection(
                websocket=websocket,
                user=user,
                state=ConnectionState.CONNECTED
            )
            self._clients[websocket] = client
            logger.info(f"Client connected: {user.get('username', 'unknown')}")
            return client
    
    async def disconnect_client(self, websocket: WebSocket):
        """Remove a UI client connection"""
        async with self._lock:
            if websocket in self._clients:
                client = self._clients.pop(websocket)
                logger.info(f"Client disconnected: {client.user.get('username', 'unknown')}")
    
    def get_clients(self) -> Dict[WebSocket, ClientConnection]:
        """Get all connected clients"""
        return self._clients.copy()
    
    # ==================== Command System ====================
    
    async def send_node_command(self, node_id: str, command: str, data: dict = None, timeout: float = 60.0) -> Optional[Dict]:
        """Send a command to a node and wait for response"""
        if node_id not in self._node_sockets:
            logger.warning(f"Cannot send command to {node_id}: no WebSocket connection")
            return None
        
        task_id = f"task_{secrets.token_hex(4)}"
        msg = {"task_id": task_id, "command": command, "data": data or {}}
        
        future = asyncio.get_event_loop().create_future()
        self._pending_tasks[task_id] = future
        self._pending_task_nodes[task_id] = node_id
        
        try:
            await self._node_sockets[node_id].send_json(msg)
            logger.debug(f"Sent command {command} to {node_id} (task: {task_id})")
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Command {command} to {node_id} timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to send command to {node_id}: {e}")
            return None
        finally:
            # ensure mapping cleaned up
            self._pending_tasks.pop(task_id, None)
            self._pending_task_nodes.pop(task_id, None)
    
    def complete_task(self, task_id: str, result: Any):
        """Complete a pending task with its result"""
        if task_id in self._pending_tasks:
            future = self._pending_tasks.pop(task_id)
            # remove node mapping if present
            self._pending_task_nodes.pop(task_id, None)
            if not future.done():
                future.set_result(result)
    
    # ==================== Broadcasting ====================
    
    async def broadcast_to_clients(self, force: bool = False):
        """Broadcast current state to all connected clients"""
        now = time.time()
        
        # Throttle broadcasts unless forced
        if not force and now - self._last_broadcast < self._broadcast_throttle:
            return
        
        self._last_broadcast = now
        
        # Prune stale nodes first
        await self.prune_stale_nodes()

        if not self._clients:
            return

        # Prepare per-client summaries then send concurrently to avoid slow sequential sends
        send_coros = []
        for websocket, client in list(self._clients.items()):
            # build filtered view for this client
            filtered_nodes = {}
            total_user_browsers = 0

            for nid, node in self._nodes.items():
                if now - node.last_seen > self._node_timeout:
                    continue

                filtered_browsers = [
                    {
                        "id": b.get("id"),
                        "mode": b.get("mode", "ephemeral"),
                        "profile_id": b.get("profile_id"),
                        "owner": b.get("owner")
                    }
                    for b in node.browsers
                    if isinstance(b, dict) and b.get("owner") == client.user["username"]
                ]

                user_browser_count = len(filtered_browsers)
                total_user_browsers += user_browser_count

                filtered_nodes[nid] = {
                    "url": node.url,
                    "browsers_count": user_browser_count,
                    "browsers": filtered_browsers,
                    "status": node.status,
                    "state": node.state.name,
                    "last_seen": node.last_seen
                }

            summary = {
                "type": "update",
                "nodes": filtered_nodes,
                "total_browsers": total_user_browsers,
                "total_nodes": len([n for n in filtered_nodes.values() if n["browsers_count"] > 0]),
                "timestamp": now
            }

            send_coros.append(self._send_to_client(websocket, client, summary))

        # Run all sends concurrently and allow them to handle their own errors
        if send_coros:
            await asyncio.gather(*send_coros, return_exceptions=True)

    async def _send_to_client(self, websocket: WebSocket, client: ClientConnection, summary: Dict[str, Any]):
        try:
            await websocket.send_json(summary)
            client.messages_sent += 1
            client.last_activity = time.time()
        except Exception as e:
            logger.error(f"Failed to broadcast to client: {e}")
            await self.disconnect_client(websocket)
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        now = time.time()
        active_nodes = sum(1 for n in self._nodes.values() if now - n.last_seen < self._node_timeout)
        total_browsers = sum(n.browsers_count for n in self._nodes.values() if now - n.last_seen < self._node_timeout)
        
        return {
            "total_nodes": len(self._nodes),
            "active_nodes": active_nodes,
            "total_browsers": total_browsers,
            "connected_clients": len(self._clients),
            "pending_tasks": len(self._pending_tasks)
        }

# Global connection manager instance
connection_manager = ConnectionManager()

# ======================================================================
# DATABASE AND AUTHENTICATION
# ======================================================================

DB_PATH = "hub_data.db"
pwd_context = CryptContext(schemes=["pbkdf2_sha256", "bcrypt"], deprecated="auto")
app = FastAPI(title="StealthNode Hub")

# Session configuration
SESSION_TTL = int(os.getenv("SESSION_TTL", 60 * 60 * 24))  # 24 hours default

# Session store (Redis-backed with in-memory fallback)
class InMemorySessionStore:
    def __init__(self):
        self._store: Dict[str, tuple] = {}
        self._lock = asyncio.Lock()

    async def set(self, sid: str, username: str, ttl: int = SESSION_TTL):
        expires = time.time() + ttl
        async with self._lock:
            self._store[sid] = (username, expires)

    async def get(self, sid: str) -> Optional[str]:
        async with self._lock:
            v = self._store.get(sid)
            if not v:
                return None
            username, expires = v
            if time.time() > expires:
                del self._store[sid]
                return None
            return username

    async def delete(self, sid: str):
        async with self._lock:
            self._store.pop(sid, None)

    async def exists(self, sid: str) -> bool:
        return (await self.get(sid)) is not None


class RedisSessionStore:
    def __init__(self, url: str = None):
        self._url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = None

    async def connect(self):
        if aioredis is None:
            raise RuntimeError("redis.asyncio not available")
        self._client = aioredis.from_url(self._url)

    async def set(self, sid: str, username: str, ttl: int = SESSION_TTL):
        await self._client.set(sid, username, ex=ttl)

    async def get(self, sid: str) -> Optional[str]:
        v = await self._client.get(sid)
        if v is None:
            return None
        if isinstance(v, bytes):
            return v.decode('utf-8')
        return str(v)

    async def delete(self, sid: str):
        await self._client.delete(sid)

    async def exists(self, sid: str) -> bool:
        return bool(await self._client.exists(sid))


# global session store instance; initialized on startup
session_store = None

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
    
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN browser_limit INTEGER DEFAULT 10")
    except sqlite3.OperationalError:
        pass
    
    admin_pass = "joemake"
    cursor.execute("SELECT password_hash FROM users WHERE username = 'admin'")
    row = cursor.fetchone()
    
    if not row:
        cursor.execute("INSERT INTO users (username, password_hash, is_admin, browser_limit) VALUES (?, ?, ?, ?)",
                       ("admin", pwd_context.hash(admin_pass), 1, -1))
    else:
        try:
            pwd_context.identify(row[0])
        except:
            cursor.execute("UPDATE users SET password_hash = ? WHERE username = 'admin'",
                           (pwd_context.hash(admin_pass),))
    
    conn.commit()
    # Create table to track generated node builds
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS node_builds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            node_id TEXT UNIQUE,
            token TEXT,
            filename TEXT,
            created_at REAL
        )
    """)
    conn.commit()
    conn.close()

@app.on_event("startup")
def on_startup():
    init_db()
    global session_store
    # Prefer Redis if available
    if aioredis is not None:
        try:
            rs = RedisSessionStore()
            asyncio.get_event_loop().run_until_complete(rs.connect())
            session_store = rs
            logger.info("Using Redis session store")
        except Exception as e:
            logger.warning(f"Redis session init failed, falling back to memory store: {e}")
            session_store = InMemorySessionStore()
    else:
        session_store = InMemorySessionStore()

    logger.info("StealthNode Hub started")

active_sessions: Dict[str, str] = {}

class UserCreate(BaseModel):
    username: str
    password: str
    is_admin: bool = False
    browser_limit: int = 10

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


async def run_db_query(func, *args, **kwargs):
    """Run a blocking DB function in the default threadpool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Legacy compatibility - keep old nodes dict as reference to connection_manager
NODE_TIMEOUT = 15

class NodeInfo(BaseModel):
    node_id: str
    url: str
    browsers_count: int
    browsers: Optional[List[Any]] = None
    status: str = "healthy"
    error: Optional[str] = None
    health: Optional[Dict[str, Any]] = None
    token: Optional[str] = None

async def get_current_user(session_id: Optional[str] = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")

    username = await session_store.get(session_id)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not logged in")
    def _query(u):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM users WHERE username = ?", (u,)).fetchone()
        conn.close()
        return dict(row) if row else None

    user = await run_db_query(_query, username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user

async def get_admin_user(current_user: sqlite3.Row = Depends(get_current_user)):
    if not current_user["is_admin"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user

# ======================================================================
# API ENDPOINTS
# ======================================================================

@app.post("/register")
async def register_node(info: NodeInfo):
    """Nodes call this to announce presence. Uses ConnectionManager for state tracking."""
    # Validate token if a build entry exists for this node_id
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute('SELECT token FROM node_builds WHERE node_id = ?', (info.node_id,)).fetchone()
        conn.close()
        if row:
            expected = row['token']
            if not info.token or info.token != expected:
                raise HTTPException(status_code=403, detail='Invalid node token')
    except HTTPException:
        raise
    except Exception:
        # If DB lookup fails, continue but log
        logger.warning('Failed to validate node token')
    node, changed = await connection_manager.register_node(
        node_id=info.node_id,
        url=info.url,
        browsers=info.browsers,
        browsers_count=info.browsers_count,
        status=info.status,
        error=info.error,
        health=info.health
    )
    
    if changed:
        asyncio.create_task(connection_manager.broadcast_to_clients(force=True))
    
    return {"status": "ok", "state": node.state.name}

@app.websocket("/ws/hub")
async def hub_endpoint(websocket: WebSocket, session_id: Optional[str] = Cookie(None)):
    """WebSocket endpoint for UI clients"""
    if not session_id:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    username = await session_store.get(session_id)
    if not username:
        await websocket.close(code=4001, reason="Unauthorized")
        return
    def _get_user(u):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        r = conn.execute("SELECT * FROM users WHERE username = ?", (u,)).fetchone()
        conn.close()
        return dict(r) if r else None

    user_row = await run_db_query(_get_user, username)
    if not user_row:
        await websocket.close(code=4001, reason="User not found")
        return

    user_info = {"username": username, "is_admin": bool(user_row.get("is_admin"))}
    await websocket.accept()
    
    # Register with connection manager
    client = await connection_manager.connect_client(websocket, user_info)
    
    # Send immediate state update
    await connection_manager.broadcast_to_clients(force=True)
    
    try:
        while True:
            data = await websocket.receive_text()
            client.messages_received += 1
            client.last_activity = time.time()
            
            # Handle client messages (ping/pong, commands, etc.)
            try:
                msg = __import__('json').loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})
            except:
                pass
                
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.disconnect_client(websocket)


# ==================== Node build management (admin) ====================

def _create_node_package(name: Optional[str] = None) -> Dict[str, Any]:
    """Create a ZIP package containing `node.py`, a run script and config. Returns metadata."""
    import zipfile, shutil
    builds_dir = BASE_DIR / 'builds'
    builds_dir.mkdir(exist_ok=True)

    node_src = BASE_DIR / 'node.py'
    if not node_src.exists():
        raise RuntimeError('node.py not found on server')

    build_name = name or f'node-{secrets.token_hex(4)}'
    node_id = f'node-{secrets.token_hex(6)}'
    token = secrets.token_urlsafe(16)
    timestamp = time.time()

    folder = builds_dir / build_name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()

    # Copy node source
    shutil.copy2(str(node_src), str(folder / 'node.py'))

    # Write a small env file and run scripts
    env_text = f"""
SERVER_URL={os.getenv('SERVER_URL','http://localhost:7860')}
NODE_ID={node_id}
NODE_TOKEN={token}
""".strip()
    (folder / 'node.env').write_text(env_text)

    # Windows launcher
    bat = f"""
@echo off
set SERVER_URL={os.getenv('SERVER_URL','http://localhost:7860')}
set NODE_ID={node_id}
set NODE_TOKEN={token}
python node.py
"""
    (folder / 'run_node.bat').write_text(bat)

    # Unix launcher
    sh = f"""
#!/bin/sh
export SERVER_URL={os.getenv('SERVER_URL','http://localhost:7860')}
export NODE_ID={node_id}
export NODE_TOKEN={token}
python3 node.py
"""
    (folder / 'run_node.sh').write_text(sh)
    os.chmod(str(folder / 'run_node.sh'), 0o755)

    # Copy requirements for user to install
    try:
        shutil.copy2(str(BASE_DIR / 'requirements.txt'), str(folder / 'requirements.txt'))
    except Exception:
        pass

    # Create ZIP
    zip_path = builds_dir / f"{build_name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as z:
        for f in folder.iterdir():
            z.write(str(f), arcname=f.name)

    # Clean up folder
    shutil.rmtree(folder)

    # Store metadata in DB
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO node_builds (name, node_id, token, filename, created_at) VALUES (?, ?, ?, ?, ?)",
                   (build_name, node_id, token, str(zip_path.name), timestamp))
    conn.commit()
    build_id = cursor.lastrowid
    conn.close()

    return {"id": build_id, "name": build_name, "node_id": node_id, "token": token, "filename": zip_path.name, "created_at": timestamp}


@app.get('/api/admin/builds')
async def list_builds(admin: sqlite3.Row = Depends(get_admin_user)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT id, name, node_id, token, filename, created_at FROM node_builds ORDER BY created_at DESC').fetchall()
    conn.close()
    builds = []
    builds_dir = BASE_DIR / 'builds'
    for r in rows:
        d = dict(r)
        filename = d.get('filename')
        size = None
        if filename:
            try:
                p = builds_dir / filename
                if p.exists():
                    size = p.stat().st_size
            except Exception:
                size = None
        d['size'] = size
        builds.append(d)
    return builds


@app.post('/api/admin/builds/{build_id}/regenerate-token')
async def regenerate_build_token(build_id: int, admin: sqlite3.Row = Depends(get_admin_user)):
    """Generate a new token for an existing build and return it."""
    new_token = secrets.token_urlsafe(16)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE node_builds SET token = ? WHERE id = ?', (new_token, build_id))
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail='Build not found')
    conn.commit()
    conn.close()
    return {"status": "ok", "id": build_id, "token": new_token}


@app.post('/api/admin/builds/{build_id}/replace')
async def replace_build_file(build_id: int, file: UploadFile = File(...), admin: sqlite3.Row = Depends(get_admin_user)):
    """Replace the file for an existing build (keeps display name)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT filename FROM node_builds WHERE id = ?', (build_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail='Build not found')
    builds_dir = BASE_DIR / 'builds'
    builds_dir.mkdir(exist_ok=True)

    # save uploaded file; replace existing file name to keep DB reference
    existing_filename = row['filename']
    target_path = builds_dir / existing_filename
    try:
        content = await file.read()
        with open(target_path, 'wb') as fh:
            fh.write(content)
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f'Failed to replace file: {e}')

    # update created_at to now
    now = time.time()
    cursor = conn.cursor()
    cursor.execute('UPDATE node_builds SET created_at = ? WHERE id = ?', (now, build_id))
    conn.commit()
    conn.close()
    return {"status": "ok", "id": build_id, "filename": existing_filename, "created_at": now}


@app.get('/api/admin/builds/{build_id}/meta')
async def build_meta(build_id: int, admin: sqlite3.Row = Depends(get_admin_user)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT id, name, node_id, token, filename, created_at FROM node_builds WHERE id = ?', (build_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail='Build not found')
    data = dict(row)
    builds_dir = BASE_DIR / 'builds'
    filename = data.get('filename')
    if filename:
        p = builds_dir / filename
        if p.exists():
            try:
                data['size'] = p.stat().st_size
            except Exception:
                data['size'] = None
    return data


@app.post('/api/admin/builds')
async def create_build(req: dict, admin: sqlite3.Row = Depends(get_admin_user)):
    name = req.get('name') if isinstance(req, dict) else None
    try:
        meta = await run_db_query(_create_node_package, name)
        return {"status": "ok", "message": f"Build created: {meta['filename']}", "id": meta['id']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/admin/builds/upload')
async def upload_build(file: UploadFile = File(...), name: Optional[str] = Form(None), admin: sqlite3.Row = Depends(get_admin_user)):
    """Upload an already-built package (zip/exe) and register it as a build with generated node token."""
    builds_dir = BASE_DIR / 'builds'
    builds_dir.mkdir(exist_ok=True)

    # save uploaded file
    filename = (name or secrets.token_hex(6)) + '-' + os.path.basename(file.filename)
    safe_path = builds_dir / filename
    try:
        with open(safe_path, 'wb') as fh:
            content = await file.read()
            fh.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # create node id and token for this build
    node_id = f'node-{secrets.token_hex(6)}'
    token = secrets.token_urlsafe(16)
    created_at = time.time()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO node_builds (name, node_id, token, filename, created_at) VALUES (?, ?, ?, ?, ?)",
                   (name or filename, node_id, token, filename, created_at))
    conn.commit()
    build_id = cursor.lastrowid
    conn.close()

    return {"status": "ok", "id": build_id, "name": name or filename, "node_id": node_id, "token": token, "filename": filename, "created_at": created_at}


@app.put('/api/admin/builds/{build_id}')
async def rename_build(build_id: int, payload: Dict[str, Any], admin: sqlite3.Row = Depends(get_admin_user)):
    """Rename a build (change display name). Does not change filename."""
    new_name = payload.get('name') if isinstance(payload, dict) else None
    if not new_name:
        raise HTTPException(status_code=400, detail='Missing name')
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('UPDATE node_builds SET name = ? WHERE id = ?', (new_name, build_id))
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail='Build not found')
    conn.commit()
    conn.close()
    return {"status": "ok", "id": build_id, "name": new_name}


@app.get('/api/admin/builds/download/{build_id}')
async def download_build(build_id: int, admin: sqlite3.Row = Depends(get_admin_user)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT filename FROM node_builds WHERE id = ?', (build_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail='Build not found')
    builds_dir = BASE_DIR / 'builds'
    path = builds_dir / row['filename']
    if not path.exists():
        raise HTTPException(status_code=404, detail='Build file missing')
    return FileResponse(path, media_type='application/zip', filename=row['filename'])


@app.get('/api/admin/builds/latest')
async def get_latest_build():
    """Returns metadata for the most recently created build."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT id, filename FROM node_builds ORDER BY created_at DESC LIMIT 1').fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail='No builds available')
    return dict(row)

@app.get('/download-latest')
async def download_latest_node():
    """Redirects to the latest build download."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT id FROM node_builds ORDER BY created_at DESC LIMIT 1').fetchone()
    conn.close()
    if not row:
        return HTMLResponse("No builds available yet. Please use the Admin panel to generate or upload one.")
    return RedirectResponse(url=f"/api/admin/builds/download/{row['id']}")

@app.delete("/api/admin/builds/{build_id}")
async def delete_build(build_id: int, admin: sqlite3.Row = Depends(get_admin_user)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute('SELECT filename FROM node_builds WHERE id = ?', (build_id,)).fetchone()
    if row:
        try:
            path = BASE_DIR / 'builds' / row['filename']
            if path.exists():
                path.unlink()
        except:
            pass
    cursor = conn.cursor()
    cursor.execute('DELETE FROM node_builds WHERE id = ?', (build_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}
@app.get("/api/profiles")
async def get_all_profiles(current_user: sqlite3.Row = Depends(get_current_user)):
    """Fetch profiles from all active nodes"""
    all_profiles = {}
    active_nodes = connection_manager.get_active_nodes()
    
    for nid, node in active_nodes.items():
        res = await connection_manager.send_node_command(nid, "get_profiles")
        if res and "profiles" in res:
            all_profiles[nid] = res["profiles"]
            continue
            
        # Fallback to legacy HTTP
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{node.url}/profiles")
                if resp.status_code == 200:
                    all_profiles[nid] = resp.json().get("profiles", [])
        except:
            pass
    
    return all_profiles

@app.websocket("/ws/node/{node_id}")
async def node_command_endpoint(websocket: WebSocket, node_id: str):
    """WebSocket used by nodes to receive commands from the hub"""
    await websocket.accept()
    await connection_manager.connect_node_websocket(node_id, websocket)
    
    logger.info(f"Node {node_id} command channel established")
    
    try:
        while True:
            data = await websocket.receive_json()
            task_id = data.get("task_id")
            if task_id:
                connection_manager.complete_task(task_id, data.get("result"))
    except WebSocketDisconnect:
        logger.info(f"Node {node_id} command channel closed")
    except Exception as e:
        logger.error(f"Node {node_id} command channel error: {e}")
    finally:
        await connection_manager.disconnect_node_websocket(node_id)

@app.delete("/api/delete_profile/{node_id}/{profile_id:path}")
async def delete_remote_profile(node_id: str, profile_id: str, current_user: sqlite3.Row = Depends(get_current_user)):
    node = connection_manager.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    result = await connection_manager.send_node_command(node_id, "delete_profile", {
        "profile_id": profile_id,
        "owner": current_user["username"],
        "is_admin": bool(current_user["is_admin"])
    })
    if result: 
        return result
    
    # Fallback to legacy HTTP
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.delete(f"{node.url}/profiles/{profile_id}")
        return resp.json()

@app.post("/api/refresh_node/{node_id}")
async def refresh_node(node_id: str, current_user: sqlite3.Row = Depends(get_current_user)):
    """Proxies a refresh request to a specific node"""
    node = connection_manager.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    result = await connection_manager.send_node_command(node_id, "refresh")
    if result: 
        return result

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{node.url}/refresh")
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/restart_node/{node_id}")
async def restart_node_action(node_id: str, admin: sqlite3.Row = Depends(get_admin_user)):
    """Proxies a restart request to a specific node (Admin only)"""
    node = connection_manager.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    result = await connection_manager.send_node_command(node_id, "restart")
    if result: 
        return result

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{node.url}/restart")
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/close_browser/{node_id}/{browser_id}")
async def close_node_browser(node_id: str, browser_id: str, current_user: sqlite3.Row = Depends(get_current_user)):
    node = connection_manager.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Check ownership unless admin
    browser = None
    browser_index = -1
    for i, b in enumerate(node.browsers):
        if isinstance(b, dict) and b.get("id") == browser_id:
            browser = b
            browser_index = i
            break
        elif b == browser_id:
            browser = {"id": b, "owner": "system"}
            browser_index = i
            break

    if not browser:
        raise HTTPException(status_code=404, detail="Browser not found on this node")
    
    is_admin = bool(current_user["is_admin"])
    if not is_admin and browser.get("owner") != current_user["username"]:
        raise HTTPException(status_code=403, detail="Unauthorized: You do not own this browser")

    result = await connection_manager.send_node_command(node_id, "close_browser", {"browser_id": browser_id})
    
    # Immediately update local state
    if browser_index >= 0:
        try:
            node.browsers.pop(browser_index)
            node.browsers_count = len(node.browsers)
            asyncio.create_task(connection_manager.broadcast_to_clients(force=True))
        except (IndexError, KeyError):
            pass
    
    if result:
        return result
    
    # Fallback to direct HTTP
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(f"{node.url}/close/{browser_id}")
        return resp.json()

# ======================================================================
# BROWSER REQUEST SYSTEM
# ======================================================================

BATCH_SIZE = 20

async def create_browser_on_node(node_id: str, mode: str, profile_id: str, owner: str):
    """Helper to create a single browser on a node"""
    cmd_result = await connection_manager.send_node_command(node_id, "create", {
        "mode": mode, 
        "profile_id": profile_id,
        "owner": owner
    })
    if cmd_result:
        return cmd_result

    # Legacy HTTP Fallback
    node = connection_manager.get_node(node_id)
    if node:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {"mode": mode, "profile_id": profile_id, "owner": owner}
                resp = await client.post(f"{node.url}/create", json=payload)
                if resp.status_code == 200:
                    return resp.json()
        except Exception as e: 
            logger.error(f"Failed to create browser on {node_id}: {e}")
    return None

def get_user_browser_count(username: str) -> int:
    """Count browsers currently owned by a user across all active nodes"""
    count = 0
    for node in connection_manager.get_active_nodes().values():
        for browser in node.browsers:
            if isinstance(browser, dict) and browser.get("owner") == username:
                count += 1
    return count

@app.get("/api/request_browser")
async def request_browser(count: int = 1, mode: str = "ephemeral", profile_id: str = None, current_user: sqlite3.Row = Depends(get_current_user)):
    """Allocates multiple browsers across the cluster"""
    username = current_user["username"]
    browser_limit = current_user["browser_limit"] if "browser_limit" in current_user.keys() else 10
    is_admin = bool(current_user["is_admin"])
    
    # Enforce browser limit
    if browser_limit != -1 and not is_admin:
        current_count = get_user_browser_count(username)
        available = max(0, browser_limit - current_count)
        if available == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Browser limit reached. You have {current_count}/{browser_limit} browsers. Close some to request more."
            )
        count = min(count, available)
    
    if count < 1:
        count = 1
    elif count > 100000:
        count = 100000
    
    active_nodes = connection_manager.get_active_nodes()
    active_node_ids = list(active_nodes.keys())
    
    if not active_node_ids:
        raise HTTPException(status_code=503, detail="No active nodes available")
    
    results = []
    failed = 0
    
    for batch_start in range(0, count, BATCH_SIZE):
        batch_count = min(BATCH_SIZE, count - batch_start)
        
        tasks = []
        for i in range(batch_count):
            sorted_nodes = sorted(active_node_ids, key=lambda nid: active_nodes[nid].browsers_count + len([t for t in tasks if t[1] == nid]))
            best_node = sorted_nodes[0]
            
            task = asyncio.create_task(
                create_browser_on_node(best_node, mode, profile_id, username)
            )
            tasks.append((task, best_node))
        
        batch_results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)
        
        for (task, node_id), result in zip(tasks, batch_results):
            if isinstance(result, Exception):
                failed += 1
            elif isinstance(result, dict) and "id" in result:
                results.append(result)
                
                node = connection_manager.get_node(node_id)
                if node:
                    existing_ids = {
                        b.get("id") if isinstance(b, dict) else b
                        for b in node.browsers
                    }
                    if result.get("id") not in existing_ids:
                        browser_entry = {
                            "id": result.get("id"),
                            "mode": result.get("mode", "ephemeral"),
                            "profile_id": result.get("profile_id"),
                            "owner": username
                        }
                        node.browsers.append(browser_entry)
                        node.browsers_count = len(node.browsers)
            else:
                failed += 1
    
    asyncio.create_task(connection_manager.broadcast_to_clients(force=True))
    
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
    active_nodes = {}
    
    for nid, node in connection_manager.get_active_nodes().items():
        filtered_browsers = [b for b in node.browsers if isinstance(b, dict) and b.get("owner") == current_user["username"]]

        active_nodes[nid] = {
            "url": node.url,
            "browsers_count": len(filtered_browsers),
            "browsers": filtered_browsers,
            "status": node.status,
            "state": node.state.name,
            "last_seen": node.last_seen
        }

    return {"nodes": active_nodes, "count": len(active_nodes)}

@app.get("/api/connection_stats")
async def get_connection_stats(current_user: sqlite3.Row = Depends(get_current_user)):
    """Get connection statistics for monitoring"""
    stats = connection_manager.get_stats()
    stats["user_browsers"] = get_user_browser_count(current_user["username"])
    return stats

# ======================================================================
# AUTHENTICATION ROUTES
# ======================================================================

@app.post("/auth/login")
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    def _get_user(u):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        r = conn.execute("SELECT * FROM users WHERE username = ?", (u,)).fetchone()
        conn.close()
        return dict(r) if r else None

    user = await run_db_query(_get_user, username)

    if not user or not pwd_context.verify(password, user.get("password_hash", "")):
        return HTMLResponse(content="""
            <script>
                alert('Invalid username or password');
                window.location.href = '/login';
            </script>
        """, status_code=401)
    
    session_id = secrets.token_hex(16)
    await session_store.set(session_id, username, SESSION_TTL)
    
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return response

@app.get("/auth/logout")
async def logout(response: Response, session_id: Optional[str] = Cookie(None)):
    if session_id:
        await session_store.delete(session_id)
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
    """Get detailed list of all browsers owned by the current user"""
    username = current_user["username"]
    my_browsers = []
    
    for nid, node in connection_manager.get_active_nodes().items():
        for browser in node.browsers:
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
    def _list():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, username, is_admin, browser_limit FROM users").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    users = await run_db_query(_list)
    result = []
    for u in users:
        user_dict = dict(u)
        user_dict["browsers_active"] = get_user_browser_count(u["username"])
        result.append(user_dict)
    return result

@app.post("/api/users")
async def create_user(user: UserCreate, admin: sqlite3.Row = Depends(get_admin_user)):
    def _create(u, pw, admin_flag, limit):
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, is_admin, browser_limit) VALUES (?, ?, ?, ?)",
                (u, pw, admin_flag, limit)
            )
            conn.commit()
        finally:
            conn.close()

    try:
        await run_db_query(_create, user.username, pwd_context.hash(user.password), 1 if user.is_admin else 0, user.browser_limit)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"status": "ok"}

@app.put("/api/users/{user_id}")
async def update_user(user_id: int, user: UserUpdate, admin: sqlite3.Row = Depends(get_admin_user)):
    """Update a user's properties"""
    def _get_and_update(uid, password, is_admin_val, browser_limit_val):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        existing = conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        if not existing:
            conn.close()
            return None

        updates = []
        params = []

        if password is not None:
            updates.append("password_hash = ?")
            params.append(pwd_context.hash(password))

        if is_admin_val is not None:
            updates.append("is_admin = ?")
            params.append(1 if is_admin_val else 0)

        if browser_limit_val is not None:
            updates.append("browser_limit = ?")
            params.append(browser_limit_val)

        if updates:
            params.append(uid)
            conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
            conn.commit()

        conn.close()
        return True

    res = await run_db_query(_get_and_update, user_id, user.password, user.is_admin, user.browser_limit)
    if res is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "ok"}

@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int, admin: sqlite3.Row = Depends(get_admin_user)):
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    def _delete(uid):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        existing = conn.execute("SELECT id FROM users WHERE id = ?", (uid,)).fetchone()
        if not existing:
            conn.close()
            return None
        conn.execute("DELETE FROM users WHERE id = ?", (uid,))
        conn.commit()
        conn.close()
        return True

    res = await run_db_query(_delete, user_id)
    if res is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "ok"}

# ======================================================================
# PAGE ROUTES
# ======================================================================

@app.get("/login", response_class=HTMLResponse)
async def login_page(session_id: Optional[str] = Cookie(None)):
    if session_id and await session_store.get(session_id):
        return RedirectResponse(url="/dashboard")
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

@app.get("/admin/builds", response_class=HTMLResponse)
async def builds_management_page(admin: sqlite3.Row = Depends(get_admin_user)):
    html_path = STATIC_DIR / "pages" / "builds.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse(content="Node builds management page missing")

@app.get("/", response_class=HTMLResponse)
async def home_page():
    """Public hero page for the root domain"""
    html_path = STATIC_DIR / "pages" / "hero.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse(content="Hero page missing. Please check static/pages/hero.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(session_id: Optional[str] = Cookie(None)):
    if not session_id or not await session_store.get(session_id):
        return RedirectResponse(url="/login")
    html_path = STATIC_DIR / "pages" / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse(content=get_fallback_html("Command Center"), status_code=200)

@app.get("/vault", response_class=HTMLResponse)
async def vault_page(session_id: Optional[str] = Cookie(None)):
    if not session_id or not await session_store.get(session_id):
        return RedirectResponse(url="/login")
    html_path = STATIC_DIR / "pages" / "vault.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse(content=get_fallback_html("Profile Vault"), status_code=200)

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(session_id: Optional[str] = Cookie(None)):
    if not session_id or not await session_store.get(session_id):
        return RedirectResponse(url="/login")
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
            h1 {{ color: #ef4444; margin-bottom: 16px; }}
            p {{ color: #94a3b8; line-height: 1.6; }}
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
            <p>The UI static files were not found. Make sure you have the 
                <code>static/</code> directory with all CSS, JS, and HTML files.</p>
            <p>Expected path: <code>{STATIC_DIR}</code></p>
        </div>
    </body>
    </html>
    """

# ======================================================================
# HEALTH CHECK
# ======================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint with connection statistics"""
    stats = connection_manager.get_stats()
    return {
        "status": "healthy",
        "active_nodes": stats["active_nodes"],
        "total_browsers": stats["total_browsers"],
        "connected_clients": stats["connected_clients"],
        "pending_tasks": stats["pending_tasks"]
    }

if __name__ == "__main__":
    print("""
    ============================================================
    
         STEALTHNODE COMMAND CENTER v2.1
         Professional Connection Management
    
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
