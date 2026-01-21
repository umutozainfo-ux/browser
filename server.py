import secrets
import time
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ======================================================================
# STEALTHNODE HUB - MODERN MULTI-PAGE ARCHITECTURE
# ======================================================================

USER_PASS = "passme"
app = FastAPI(title="StealthNode Hub")
security = HTTPBasic()

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

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    if secrets.compare_digest(credentials.password, USER_PASS):
        return "admin"
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )

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
async def get_all_profiles(username: str = Depends(get_current_username)):
    """Fetch profiles from all active nodes"""
    all_profiles = {}
    now = time.time()
    active_nodes = {nid: data for nid, data in nodes.items() if now - data["last_seen"] < NODE_TIMEOUT}
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        tasks = [client.get(f"{node['url']}/profiles") for node in active_nodes.values()]
        resps = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (nid, node), resp in zip(active_nodes.items(), resps):
            if isinstance(resp, httpx.Response) and resp.status_code == 200:
                all_profiles[nid] = resp.json().get("profiles", [])
    
    return all_profiles

@app.delete("/api/delete_profile/{node_id}/{profile_id:path}")
async def delete_remote_profile(node_id: str, profile_id: str, username: str = Depends(get_current_username)):
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node_url = nodes[node_id]["url"]
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.delete(f"{node_url}/profiles/{profile_id}")
        return resp.json()

@app.post("/api/refresh_node/{node_id}")
async def refresh_node(node_id: str, username: str = Depends(get_current_username)):
    """Proxies a refresh request to a specific node"""
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node_url = nodes[node_id]["url"]
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.post(f"{node_url}/refresh")
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/request_browser")
async def request_browser(count: int = 1, mode: str = "ephemeral", profile_id: str = None, username: str = Depends(get_current_username)):
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
async def get_nodes(username: str = Depends(get_current_username)):
    """Get current node status"""
    now = time.time()
    active_nodes = {nid: data for nid, data in nodes.items() if now - data["last_seen"] < NODE_TIMEOUT}
    return {"nodes": active_nodes, "count": len(active_nodes)}

# ======================================================================
# PAGE ROUTES - MULTI-PAGE ARCHITECTURE
# ======================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Command Center - Main browser control page"""
    html_path = STATIC_DIR / "pages" / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse(content=get_fallback_html("Command Center"), status_code=200)

@app.get("/vault", response_class=HTMLResponse)
async def vault_page():
    """Profile Vault - Manage persistent profiles"""
    html_path = STATIC_DIR / "pages" / "vault.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse(content=get_fallback_html("Profile Vault"), status_code=200)

@app.get("/settings", response_class=HTMLResponse)
async def settings_page():
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
    print(f"    Starting server at http://0.0.0.0:8000")
    print(f"    Static files: {STATIC_DIR}")
    print(f"    Login password: {USER_PASS}")
    print("""
    Pages:
      - Command Center: http://localhost:8000/
      - Profile Vault:  http://localhost:8000/vault
      - Settings:       http://localhost:8000/settings
      - Health Check:   http://localhost:8000/health
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)

