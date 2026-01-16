import secrets
import time
import httpx
from typing import Dict, Any, List, Optional
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

# Global state
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

@app.post("/api/request_browser")
async def request_browser(username: str = Depends(get_current_username)):
    now = time.time()
    active_node_ids = [nid for nid, data in nodes.items() if now - data["last_seen"] < 15]
    
    if not active_node_ids:
        raise HTTPException(status_code=500, detail="No active nodes available")
    
    # Smart logic: Pick node with minimum browser count
    best_node_id = min(active_node_ids, key=lambda nid: nodes[nid]["browsers_count"])
    node_url = nodes[best_node_id]["url"]
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{node_url}/create")
            if resp.status_code == 200:
                return resp.json()
            else:
                raise HTTPException(status_code=500, detail="Node rejected request")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")

# ======================================================================
# DASHBOARD
# ======================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    return """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StealthNode Center</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    fontFamily: {
                        sans: ['Outfit', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace'],
                    },
                    colors: {
                        primary: '#3b82f6',
                        secondary: '#6366f1',
                        accent: '#8b5cf6',
                        dark: '#020617',
                    }
                }
            }
        }
    </script>
    <style>
        body { background: #020617; color: #f8fafc; overflow-x: hidden; }
        .glass { background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(16px); border: 1px solid rgba(255,255,255,0.05); }
        .glass-card { background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.05); transition: all 0.3s ease; }
        .glass-card:hover { border-color: rgba(59, 130, 246, 0.4); transform: translateY(-4px); box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1); }
        .animated-gradient { background: linear-gradient(-45deg, #3b82f6, #6366f1, #8b5cf6, #ec4899); background-size: 400% 400%; animation: gradient 15s ease infinite; }
        @keyframes gradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        .browser-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.5rem; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; }
        .status-online { background-color: #22c55e; box-shadow: 0 0 10px #22c55e; }
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
        .selected-card { border-color: #3b82f6 !important; background: rgba(59, 130, 246, 0.1) !important; }
        .mode-toggle { display: flex; background: rgba(0,0,0,0.3); border-radius: 8px; padding: 2px; }
        .mode-btn { padding: 2px 8px; border-radius: 6px; font-size: 8px; font-weight: 800; color: #64748b; transition: all 0.2s; }
        .mode-btn.active { background: #3b82f6; color: white; }
    </style>
</head>
<body class="min-h-screen flex flex-col font-sans">
    <!-- Header -->
    <header class="glass sticky top-0 z-[100] px-6 py-4 flex items-center justify-between border-b border-white/5">
        <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-2xl animated-gradient flex items-center justify-center shadow-lg shadow-primary/20">
                <span class="text-xl">🛡️</span>
            </div>
            <div>
                <h1 class="text-xl font-extrabold tracking-tight">STEALTH<span class="text-primary">NODE</span></h1>
                <p class="text-[10px] uppercase tracking-widest text-slate-400 font-bold">Distributed Browser Network</p>
            </div>
        </div>

        <div class="flex items-center gap-4">
            <div id="stats" class="hidden md:flex items-center gap-6 px-4 py-2 rounded-2xl bg-white/5 text-xs font-mono">
                <div class="flex flex-col"><span class="text-slate-500">NODES</span><span id="nodeCount">0</span></div>
                <div class="flex flex-col"><span class="text-slate-500">BROWSERS</span><span id="browserCount">0</span></div>
            </div>
            
            <button onclick="requestNewBrowser()" id="reqBtn" class="bg-primary hover:bg-primary/80 text-white px-6 py-2.5 rounded-2xl text-sm font-bold shadow-lg shadow-primary/30 transition active:scale-95 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M12 4v16m8-8H4" /></svg>
                REQUEST BROWSER
            </button>
        </div>
    </header>

    <!-- Toolbar -->
    <div class="glass mx-6 mt-6 rounded-3xl p-4 flex flex-wrap items-center justify-between gap-4 border border-white/5">
        <div class="flex items-center gap-2">
            <button onclick="toggleControlMode('all')" id="modeAll" class="px-4 py-2 rounded-xl text-xs font-bold transition-all bg-primary text-white">CONTROL ALL</button>
            <button onclick="toggleControlMode('single')" id="modeSingle" class="px-4 py-2 rounded-xl text-xs font-bold transition-all bg-white/5 text-slate-400">CONTROL INDIVIDUAL</button>
        </div>

        <div class="flex-1 max-w-xl flex gap-2">
            <div class="relative flex-1">
                <span class="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 text-xs text-mono">URL</span>
                <input id="globalUrlInput" type="text" placeholder="https://google.com" class="w-full bg-slate-900/50 border border-white/10 rounded-2xl pl-12 pr-4 py-2 text-sm outline-none focus:border-primary/50 transition font-mono">
            </div>
            <button onclick="globalNavigate()" class="bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-2 rounded-2xl text-xs font-bold transition active:scale-95">GO ALL</button>
        </div>

        <div class="flex items-center gap-2">
            <button onclick="refreshAll()" class="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-slate-400 transition" title="Refresh All Views">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
            </button>
            <button onclick="closeAllBrowsers()" class="p-2.5 rounded-xl bg-red-500/10 hover:bg-red-500/20 text-red-500 transition" title="Kill All">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
            </button>
        </div>
    </div>

    <!-- Main Grid -->
    <main class="flex-1 p-6 custom-scrollbar overflow-y-auto">
        <div id="browserGrid" class="browser-grid pb-20">
            <!-- Loading State -->
            <div class="col-span-full py-20 flex flex-col items-center justify-center text-slate-500 opacity-50">
                <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
                <p class="text-sm font-medium">Scanning for active nodes...</p>
            </div>
        </div>
    </main>

    <!-- Extended Control View (Modal) -->
    <div id="fullViewModal" class="fixed inset-0 z-[200] hidden bg-dark/95 backdrop-blur-xl flex flex-col">
        <div class="p-4 border-b border-white/5 flex items-center justify-between">
            <div class="flex items-center gap-4">
                <div class="px-3 py-1 bg-primary/20 text-primary text-[10px] font-extrabold rounded-full tracking-widest uppercase">Live Link</div>
                <h3 id="currentBrowserTitle" class="text-sm font-bold text-slate-300">Browser Instance</h3>
            </div>
            <div class="flex items-center gap-3">
                <input id="modalUrlInput" type="text" class="bg-white/5 border border-white/10 rounded-xl px-4 py-1.5 text-xs w-96 outline-none focus:border-primary transition" placeholder="Enter URL...">
                <button onclick="modalNavigate()" class="bg-primary text-white text-[10px] font-bold px-4 py-2 rounded-xl">NAVIGATE</button>
                <div class="w-px h-6 bg-white/5 mx-2"></div>
                <button onclick="closeFullView()" class="p-2 rounded-xl bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-white transition">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
                </button>
            </div>
        </div>
        <div class="flex-1 relative flex items-center justify-center overflow-hidden bg-black p-4">
            <canvas id="mainView" class="max-w-full max-h-full shadow-2xl bg-slate-900 rounded-lg cursor-crosshair"></canvas>
            <video id="mainVideo" class="hidden max-w-full max-h-full shadow-2xl bg-slate-900 rounded-lg cursor-crosshair" autoplay playsinline></video>
            <div id="latencyIndicator" class="absolute bottom-6 right-6 px-3 py-1 bg-black/50 text-[10px] text-green-500 font-mono rounded-lg border border-green-500/20">5ms</div>
        </div>
    </div>

    <script>
        let nodes = {};
        let browserInstances = []; // [{nodeId, browserId, url, ws, canvas}]
        let controlMode = 'all'; // 'all' or 'single'
        let activeFullView = null; // {nodeId, browserId}
        
        async function fetchSystemState() {
            try {
                const res = await fetch('/api/nodes');
                if(!res.ok) return;
                const newNodes = await res.json();
                
                // Compare and update nodes
                nodes = newNodes;
                syncBrowsers();
                updateStats();
            } catch(e) {}
        }

        function updateStats() {
            document.getElementById('nodeCount').innerText = Object.keys(nodes).length;
            let total = 0;
            Object.values(nodes).forEach(n => total += n.browsers_count);
            document.getElementById('browserCount').innerText = total;
        }

        function syncBrowsers() {
            const currentIds = new Set();
            const grid = document.getElementById('browserGrid');
            
            // Build list of active browsers across all nodes
            const activeBrowsers = [];
            Object.entries(nodes).forEach(([nid, n]) => {
                n.browsers.forEach(bid => {
                    const id = nid + ':' + bid;
                    currentIds.add(id);
                    activeBrowsers.push({ nid, bid, url: n.url });
                });
            });

            // 1. Remove dead instances
            browserInstances = browserInstances.filter(inst => {
                const id = inst.nid + ':' + inst.bid;
                if (!currentIds.has(id)) {
                    if (inst.ws) inst.ws.close();
                    document.getElementById('card-' + id.replace(/:/g, '-'))?.remove();
                    return false;
                }
                return true;
            });

            // 2. Add new instances
            activeBrowsers.forEach(b => {
                const id = b.nid + ':' + b.bid;
                if (!browserInstances.find(inst => inst.nid === b.nid && inst.bid === b.bid)) {
                    addBrowserCard(b.nid, b.bid, b.url);
                }
            });

            if (browserInstances.length === 0) {
                grid.innerHTML = `
                    <div class="col-span-full py-20 flex flex-col items-center justify-center text-slate-500">
                        <p class="text-sm font-medium">No active browsers found.</p>
                        <button onclick="requestNewBrowser()" class="mt-4 text-primary text-xs font-bold hover:underline">Launch your first instance</button>
                    </div>
                `;
            } else if (grid.querySelector('.animate-spin')) {
                grid.innerHTML = ''; // Clear loading if browsers found
            }
        }

        function addBrowserCard(nid, bid, nodeUrl) {
            const safeId = (nid + '-' + bid).replace(/:/g, '-');
            const container = document.getElementById('browserGrid');
            
            const card = document.createElement('div');
            card.id = 'card-' + safeId;
            card.className = 'glass-card rounded-[2rem] overflow-hidden flex flex-col group';
            card.innerHTML = `
                <div class="p-4 flex items-center justify-between border-b border-white/5">
                    <div class="flex items-center gap-2 overflow-hidden">
                        <div class="status-dot status-online"></div>
                        <span class="text-[10px] font-bold text-slate-400 truncate">${nid}</span>
                        <span class="text-[10px] text-slate-600">/</span>
                        <span class="text-xs font-mono text-white">${bid}</span>
                    </div>
                    <div class="flex items-center gap-2">
                         <div class="mode-toggle">
                            <button id="btn-rtc-${safeId}" onclick="switchMode('${nid}', '${bid}', 'webrtc')" class="mode-btn active">RTC</button>
                            <button id="btn-ws-${safeId}" onclick="switchMode('${nid}', '${bid}', 'ws')" class="mode-btn">WS</button>
                         </div>
                         <button onclick="openFullView('${nid}', '${bid}')" class="p-1.5 hover:bg-white/10 rounded-lg text-slate-400 opacity-0 group-hover:opacity-100 transition" title="Enlarge">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M3 4a1 1 0 011-1h4a1 1 0 010 2H6.414l2.293 2.293a1 1 0 11-1.414 1.414L5 6.414V8a1 1 0 01-2 0V4zm9 1a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-2 0V6.414l-2.293 2.293a1 1 0 11-1.414-1.414L13.586 5H12zm-9 7a1 1 0 012 0v1.586l2.293-2.293a1 1 0 111.414 1.414L4.414 15H6a1 1 0 010 2H2a1 1 0 01-1-1v-4zm11-1a1 1 0 011.414 1.414L15.586 15H17a1 1 0 110 2h-4a1 1 0 01-1-1v-4a1 1 0 011-1z" clip-rule="evenodd" /></svg>
                        </button>
                        <button onclick="terminateBrowser('${nid}', '${bid}')" class="p-1.5 hover:bg-red-500/20 rounded-lg text-red-500 opacity-0 group-hover:opacity-100 transition" title="Close Instance">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
                        </button>
                    </div>
                </div>
                <div class="aspect-video bg-black relative overflow-hidden cursor-crosshair">
                    <canvas id="view-${safeId}" class="w-full h-full object-contain"></canvas>
                    <video id="video-${safeId}" class="hidden w-full h-full object-contain" autoplay playsinline></video>
                    <div class="absolute inset-0 bg-primary/5 pointer-events-none opacity-0 group-hover:opacity-100 transition"></div>
                </div>
                <div class="p-3 bg-white/5 flex gap-2">
                    <input id="input-${safeId}" type="text" placeholder="https://..." class="flex-1 bg-slate-900/50 border border-white/5 rounded-xl px-3 py-1 text-[10px] font-mono outline-none focus:border-primary/30 transition">
                    <button onclick="navigateSingle('${nid}', '${bid}')" class="bg-primary/20 text-primary text-[10px] font-bold px-3 py-1 rounded-xl hover:bg-primary hover:text-white transition">GO</button>
                </div>
            `;
            container.appendChild(card);

            const canvas = document.getElementById('view-' + safeId);
            const wsUrl = nodeUrl.replace('http', 'ws') + '/ws/' + bid;
            const ws = new WebSocket(wsUrl);
            const ctx = canvas.getContext('2d');

            const inst = { nid, bid, ws, canvas, ctx, mode: 'webrtc', video: document.getElementById('video-' + safeId) };
            browserInstances.push(inst);

            // WebRTC Logic
            setupWebRTC(inst, nodeUrl);

            let lastDraw = 0;
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'frame') {
                    const now = performance.now();
                    const isFullView = activeFullView && activeFullView.nid === nid && activeFullView.bid === bid;
                    
                    // Throttle background cards to ~10 FPS to save user CPU, but keep full view fast
                    if (!isFullView && now - lastDraw < 100) return;
                    lastDraw = now;

                    const img = new Image();
                    img.onload = () => {
                        if (canvas.width !== img.width) {
                            canvas.width = img.width;
                            canvas.height = img.height;
                        }
                        ctx.drawImage(img, 0, 0);
                        
                        // If this matches active full view, draw there too
                        if (isFullView) {
                            const mainCv = document.getElementById('mainView');
                            if (mainCv.width !== img.width) {
                                mainCv.width = img.width;
                                mainCv.height = img.height;
                            }
                            mainCv.getContext('2d').drawImage(img, 0, 0);
                        }
                    };
                    img.src = 'data:image/jpeg;base64,' + msg.data;
                }
            };

            // Event Hooks for Multi-Control (attach to BOTH canvas and video)
            const video = inst.video;
            
            // Canvas events
            canvas.onmousedown = (e) => handleCanvasEvent('mousedown', {nid, bid}, e);
            canvas.onmouseup = (e) => handleCanvasEvent('mouseup', {nid, bid}, e);
            canvas.onmousemove = (e) => handleCanvasEvent('mousemove', {nid, bid}, e);
            canvas.onclick = (e) => handleCanvasEvent('click', {nid, bid}, e);
            canvas.onwheel = (e) => handleCanvasEvent('scroll', {nid, bid}, e);
            
            // Video events (same handlers)
            video.onmousedown = (e) => handleCanvasEvent('mousedown', {nid, bid}, e);
            video.onmouseup = (e) => handleCanvasEvent('mouseup', {nid, bid}, e);
            video.onmousemove = (e) => handleCanvasEvent('mousemove', {nid, bid}, e);
            video.onclick = (e) => handleCanvasEvent('click', {nid, bid}, e);
            video.onwheel = (e) => handleCanvasEvent('scroll', {nid, bid}, e);
        }

        function handleCanvasEvent(type, target, e) {
            e.preventDefault();
            const el = e.target;
            const rect = el.getBoundingClientRect();
            
            // Get actual media resolution
            const mediaWidth = el.tagName === 'VIDEO' ? el.videoWidth : el.width;
            const mediaHeight = el.tagName === 'VIDEO' ? el.videoHeight : el.height;

            const scaleX = mediaWidth / rect.width;
            const scaleY = mediaHeight / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;

            const payload = { type, x, y, deltaY: e.deltaY };
            
            if (controlMode === 'all') {
                browserInstances.forEach(inst => {
                    if (inst.ws.readyState === WebSocket.OPEN) {
                        inst.ws.send(JSON.stringify(payload));
                    }
                });
            } else {
                const inst = browserInstances.find(i => i.nid === target.nid && i.bid === target.bid);
                if (inst && inst.ws.readyState === WebSocket.OPEN) {
                    inst.ws.send(JSON.stringify(payload));
                }
            }
        }

        async function requestNewBrowser() {
            const btn = document.getElementById('reqBtn');
            const original = btn.innerHTML;
            btn.innerHTML = '<span class="animate-spin">🔄</span> REQUESTING...';
            btn.disabled = true;

            try {
                const res = await fetch('/api/request_browser', { method: 'POST' });
                if (res.ok) {
                    await fetchSystemState();
                } else {
                    alert("Allocation failed. Check node status.");
                }
            } finally {
                btn.innerHTML = original;
                btn.disabled = false;
            }
        }

        function toggleControlMode(mode) {
            controlMode = mode;
            document.getElementById('modeAll').className = mode === 'all' 
                ? 'px-4 py-2 rounded-xl text-xs font-bold transition-all bg-primary text-white' 
                : 'px-4 py-2 rounded-xl text-xs font-bold transition-all bg-white/5 text-slate-400';
            document.getElementById('modeSingle').className = mode === 'single' 
                ? 'px-4 py-2 rounded-xl text-xs font-bold transition-all bg-primary text-white' 
                : 'px-4 py-2 rounded-xl text-xs font-bold transition-all bg-white/5 text-slate-400';
            
            // Highlight cards if in single mode
            document.querySelectorAll('.glass-card').forEach(c => c.classList.remove('selected-card'));
        }

        function globalNavigate() {
            let url = document.getElementById('globalUrlInput').value;
            if (!url) return;
            if (!url.startsWith('http')) url = 'https://' + url;
            
            browserInstances.forEach(inst => {
                if (inst.ws.readyState === WebSocket.OPEN) {
                    inst.ws.send(JSON.stringify({ type: 'navigate', url }));
                }
            });
        }

        function navigateSingle(nid, bid) {
            const safeId = (nid + '-' + bid).replace(/:/g, '-');
            let url = document.getElementById('input-' + safeId).value;
            if (!url) return;
            if (!url.startsWith('http')) url = 'https://' + url;
            
            const inst = browserInstances.find(i => i.nid === nid && i.bid === bid);
            if (inst && inst.ws.readyState === WebSocket.OPEN) {
                inst.ws.send(JSON.stringify({ type: 'navigate', url }));
            }
        }

        function openFullView(nid, bid) {
            activeFullView = { nid, bid };
            const inst = browserInstances.find(i => i.nid === nid && i.bid === bid);
            if (!inst) return;

            document.getElementById('fullViewModal').classList.remove('hidden');
            document.getElementById('currentBrowserTitle').innerText = `${nid} / ${bid}`;
            
            const mainCv = document.getElementById('mainView');
            const mainVid = document.getElementById('mainVideo');

            if (inst.mode === 'webrtc') {
                mainCv.classList.add('hidden');
                mainVid.classList.remove('hidden');
                mainVid.srcObject = inst.video.srcObject;
            } else {
                mainCv.classList.remove('hidden');
                mainVid.classList.add('hidden');
            }

            // Pipe events from main elements
            const activeEl = inst.mode === 'webrtc' ? mainVid : mainCv;
            activeEl.onmousedown = (e) => handleCanvasEvent('mousedown', activeFullView, e);
            activeEl.onmouseup = (e) => handleCanvasEvent('mouseup', activeFullView, e);
            activeEl.onmousemove = (e) => handleCanvasEvent('mousemove', activeFullView, e);
            activeEl.onwheel = (e) => handleCanvasEvent('scroll', activeFullView, e);
            
            window.onkeydown = (e) => {
                if(document.activeElement.tagName === 'INPUT') return;
                if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Tab','Backspace','Enter'].includes(e.key)) e.preventDefault();
                
                const payload = { type: 'key', key: e.key };
                if (controlMode === 'all') {
                    browserInstances.forEach(inst => {
                        if (inst.ws.readyState === WebSocket.OPEN) inst.ws.send(JSON.stringify(payload));
                    });
                } else if (activeFullView) {
                    const inst = browserInstances.find(i => i.nid === activeFullView.nid && i.bid === activeFullView.bid);
                    if (inst && inst.ws.readyState === WebSocket.OPEN) inst.ws.send(JSON.stringify(payload));
                }
            };
        }

        function closeFullView() {
            activeFullView = null;
            document.getElementById('fullViewModal').classList.add('hidden');
            window.onkeydown = null;
        }

        function modalNavigate() {
            let url = document.getElementById('modalUrlInput').value;
            if (!url || !activeFullView) return;
            if (!url.startsWith('http')) url = 'https://' + url;
            
            const inst = browserInstances.find(i => i.nid === activeFullView.nid && i.bid === activeFullView.bid);
            inst?.ws.send(JSON.stringify({ type: 'navigate', url }));
        }

        async function terminateBrowser(nid, bid) {
            const inst = browserInstances.find(i => i.nid === nid && i.bid === bid);
            if (!inst) return;
            try {
                const node = nodes[nid];
                if (node) {
                    await fetch(`${node.url}/close/${bid}`, { method: 'POST' });
                    setTimeout(fetchSystemState, 500);
                }
            } catch(e){}
        }

        async function closeAllBrowsers() {
            if (!confirm("Are you sure you want to kill ALL browser instances?")) return;
            // Best effort broadcast to all nodes would be better, but for now we iterate
            for (const node of Object.values(nodes)) {
                try { await fetch(`${node.url}/close_all`, { method: 'POST' }); } catch(e){}
            }
            setTimeout(fetchSystemState, 1000);
        }

        async function setupWebRTC(inst, nodeUrl) {
            const pc = new RTCPeerConnection();
            inst.pc = pc;

            pc.ontrack = (event) => {
                inst.video.srcObject = event.streams[0];
                if (activeFullView && activeFullView.nid === inst.nid && activeFullView.bid === inst.bid) {
                    document.getElementById('mainVideo').srcObject = event.streams[0];
                }
            };

            const offer = await pc.createOffer({ offerToReceiveVideo: true });
            await pc.setLocalDescription(offer);

            try {
                const response = await fetch(`${nodeUrl}/api/offer/${inst.bid}`, {
                    method: 'POST',
                    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
                    headers: { 'Content-Type': 'application/json' }
                });
                const answer = await response.json();
                await pc.setRemoteDescription(new RTCSessionDescription(answer));
                switchMode(inst.nid, inst.bid, 'webrtc');
            } catch (e) {
                switchMode(inst.nid, inst.bid, 'ws');
            }
        }

        function switchMode(nid, bid, mode) {
            const inst = browserInstances.find(i => i.nid === nid && i.bid === bid);
            if (!inst) return;
            inst.mode = mode;
            
            const safeId = (nid + '-' + bid).replace(/:/g, '-');
            const btnRtc = document.getElementById('btn-rtc-' + safeId);
            const btnWs = document.getElementById('btn-ws-' + safeId);
            const canvas = document.getElementById('view-' + safeId);
            const video = document.getElementById('video-' + safeId);

            if (mode === 'webrtc') {
                btnRtc.classList.add('active');
                btnWs.classList.remove('active');
                video.classList.remove('hidden');
                canvas.classList.add('hidden');
                if (activeFullView && activeFullView.nid === nid && activeFullView.bid === bid) {
                     document.getElementById('mainView').classList.add('hidden');
                     document.getElementById('mainVideo').classList.remove('hidden');
                     document.getElementById('mainVideo').srcObject = video.srcObject;
                }
            } else {
                btnWs.classList.add('active');
                btnRtc.classList.remove('active');
                canvas.classList.remove('hidden');
                video.classList.add('hidden');
                if (activeFullView && activeFullView.nid === nid && activeFullView.bid === bid) {
                     document.getElementById('mainView').classList.remove('hidden');
                     document.getElementById('mainVideo').classList.add('hidden');
                }
            }
        }

        setInterval(fetchSystemState, 3000);
        fetchSystemState();
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
