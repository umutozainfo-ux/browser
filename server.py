import os
import json
import asyncio
import time
import secrets
from typing import Dict, Any, List
from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware

# ======================================================================
# MODELS & CONFIG
# ======================================================================

USER_PASS = "passme"
app = FastAPI()
security = HTTPBasic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class NodeInfo(BaseModel):
    node_id: str
    url: str
    browsers_count: int
    browsers: List[str]

nodes: Dict[str, Dict[str, Any]] = {}

# ======================================================================
# AUTH
# ======================================================================

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    if secrets.compare_digest(credentials.password, USER_PASS):
        return "admin"
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Basic"},
    )

# ======================================================================
# API LOGIC
# ======================================================================

@app.post("/register")
async def register_node(info: NodeInfo):
    nodes[info.node_id] = {
        "url": info.url,
        "browsers_count": info.browsers_count,
        "browsers": info.browsers,
        "last_seen": time.time()
    }
    return {"status": "ok"}

@app.get("/api/nodes")
async def get_nodes(username: str = Depends(get_current_username)):
    now = time.time()
    active_nodes = {}
    for nid, data in nodes.items():
        if now - data["last_seen"] < 15:
            active_nodes[nid] = data
    return active_nodes

# ======================================================================
# DASHBOARD
# ======================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StealthNode Controller</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background: #0f172a; color: #f1f5f9; }
        .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); }
    </style>
</head>
<body class="min-h-screen">
    <nav class="glass sticky top-0 z-50 p-4 shadow-xl">
        <div class="max-w-7xl mx-auto flex justify-between items-center">
            <h1 class="text-xl font-bold tracking-tight flex items-center gap-2">
                <span class="text-blue-500">🛡️ Stealth</span>Node
            </h1>
            <div id="status" class="text-[10px] font-mono opacity-50">Connecting...</div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 py-10">
        <div id="nodesGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"></div>
    </main>

    <!-- UI Overlay for Browser -->
    <div id="browserModal" class="fixed inset-0 bg-slate-950 hidden z-[100] flex flex-col">
        <div class="p-4 flex justify-between items-center border-b border-white/5">
            <div class="flex items-center gap-4 text-sm">
                <span id="labelNode" class="text-blue-500 font-bold"></span>
                <span class="opacity-20">/</span>
                <span id="labelBrowser" class="opacity-80"></span>
            </div>
            <button onclick="closeBrowser()" class="bg-red-500/10 text-red-500 px-4 py-1 rounded-full text-xs hover:bg-red-500 hover:text-white transition">EXIT</button>
        </div>
        
        <div class="bg-slate-900 p-2 flex gap-2">
            <input id="urlInput" type="text" placeholder="google.com" class="flex-1 bg-slate-950 border border-white/10 rounded-lg px-4 py-1 text-sm outline-none">
            <button onclick="navigate()" class="bg-blue-600 px-6 py-1 rounded-lg text-sm font-bold">GO</button>
        </div>

        <div id="canvasContainer" class="flex-1 overflow-hidden relative flex items-center justify-center p-4">
            <canvas id="view" class="max-w-full max-h-full shadow-2xl bg-black cursor-crosshair"></canvas>
        </div>
    </div>

    <script>
        let nodes = {};
        let activeWs = null;
        let lastX = 0, lastY = 0;

        async function update() {
            try {
                const res = await fetch('/api/nodes');
                if(res.ok) {
                    nodes = await res.json();
                    render();
                }
            } catch(e){}
        }

        async function create(url) {
            await fetch(`${url}/create`, {method:'POST'});
            update();
        }

        function render() {
            const grid = document.getElementById('nodesGrid');
            grid.innerHTML = Object.keys(nodes).map(id => {
                const n = nodes[id];
                return `
                    <div class="glass rounded-3xl p-6 border border-white/5 shadow-lg">
                        <div class="flex justify-between mb-6">
                            <div class="font-bold">${id}</div>
                            <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        </div>
                        <div class="space-y-2">
                            ${n.browsers.map(bid => `
                                <button onclick="openB('${n.url}', '${bid}', '${id}')" class="w-full bg-slate-800 text-left px-4 py-2 rounded-xl text-xs hover:bg-blue-600 transition">Browser ${bid}</button>
                            `).join('')}
                            <button onclick="create('${n.url}')" class="w-full border border-blue-500/20 text-blue-400 py-2 rounded-xl text-xs font-bold hover:bg-blue-600 hover:text-white transition">+ NEW INSTANCE</button>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function openB(nodeUrl, bid, nid) {
            document.getElementById('browserModal').classList.remove('hidden');
            document.getElementById('labelNode').innerText = nid;
            document.getElementById('labelBrowser').innerText = bid;
            
            const wsUrl = nodeUrl.replace('http', 'ws') + '/ws/' + bid;
            const canvas = document.getElementById('view');
            const ctx = canvas.getContext('2d');
            
            if (activeWs) activeWs.close();
            activeWs = new WebSocket(wsUrl);
            
            activeWs.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'frame') {
                    const img = new Image();
                    img.onload = () => {
                        canvas.width = img.width;
                        canvas.height = img.height;
                        ctx.drawImage(img, 0, 0);
                    };
                    img.src = 'data:image/jpeg;base64,' + msg.data;
                }
            };

            // FULL MOUSE CONTROL
            canvas.onmousemove = (e) => {
                const pos = getPos(e);
                activeWs.send(JSON.stringify({ type: 'mousemove', x: pos.x, y: pos.y }));
            };
            canvas.onmousedown = (e) => {
                const pos = getPos(e);
                activeWs.send(JSON.stringify({ type: 'mousedown', x: pos.x, y: pos.y }));
            };
            canvas.onmouseup = (e) => {
                const pos = getPos(e);
                activeWs.send(JSON.stringify({ type: 'mouseup', x: pos.x, y: pos.y }));
            };
            canvas.oncontextmenu = (e) => e.preventDefault();

            // KEYBOARD
            window.onkeydown = (e) => {
                if(document.activeElement.id === 'urlInput') return;
                if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Tab','Backspace','Enter'].includes(e.key)) e.preventDefault();
                activeWs.send(JSON.stringify({ type: 'key', key: e.key }));
            };
            
            canvas.onwheel = (e) => {
                e.preventDefault();
                activeWs.send(JSON.stringify({ type: 'scroll', deltaY: e.deltaY }));
            };
        }

        function getPos(e) {
            const canvas = document.getElementById('view');
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }

        function navigate() {
            const url = document.getElementById('urlInput').value;
            if (activeWs) activeWs.send(JSON.stringify({ type: 'navigate', url }));
        }

        function closeBrowser() {
            document.getElementById('browserModal').classList.add('hidden');
            if (activeWs) activeWs.close();
        }

        setInterval(update, 3000);
        update();
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
