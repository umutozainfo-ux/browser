import secrets
import time
import asyncio
from typing import Dict, Any, List
from pydantic import BaseModel
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware

# ======================================================================
# SUPREME HUB - ARCHITECTED FOR INFINITE SCALE
# ======================================================================

USER_PASS = "passme"
app = FastAPI(title="StealthNode Hub")
security = HTTPBasic()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registry of all connected nodes
# { node_id: { url, browsers_count, browsers, last_seen } }
nodes: Dict[str, Dict[str, Any]] = {}
NODE_TIMEOUT = 15 # Seconds before a node is considered offline

class NodeInfo(BaseModel):
    node_id: str
    url: str
    browsers_count: int
    browsers: List[str]

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
        try: await client.send_json(msg)
        except: hub_clients.remove(client)

# ======================================================================
# API ENDPOINTS
# ======================================================================

@app.post("/register")
async def register_node(info: NodeInfo):
    """Nodes call this to announce presence"""
    nodes[info.node_id] = {
        "url": info.url,
        "browsers_count": info.browsers_count,
        "browsers": info.browsers,
        "last_seen": time.time()
    }
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
        if websocket in hub_clients: hub_clients.remove(websocket)

@app.get("/api/request_browser")
async def request_browser(count: int = 1, username: str = Depends(get_current_username)):
    """Allocates multiple browsers across the cluster"""
    if count < 1 or count > 10: count = 1
    
    results = []
    for _ in range(count):
        now = time.time()
        active_node_ids = [nid for nid, data in nodes.items() if now - data["last_seen"] < NODE_TIMEOUT]
        
        if not active_node_ids:
            break
        
        # Smart load balancing
        best_node_id = min(active_node_ids, key=lambda nid: nodes[nid]["browsers_count"])
        node_url = nodes[best_node_id]["url"]
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{node_url}/create")
                if resp.status_code == 200:
                    results.append(resp.json())
        except: continue
        
    return {"results": results, "count": len(results)}

# ======================================================================
# THE MASTERPIECE DASHBOARD
# ======================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StealthNode | Command Center</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    fontFamily: { sans: ['Space Grotesk', 'sans-serif'], mono: ['JetBrains Mono', 'monospace'] },
                    colors: { 
                        brand: '#6366f1',
                        surface: '#0f172a',
                        card: '#1e293b'
                    }
                }
            }
        }
    </script>
    <style>
        :root { --accent: #6366f1; --accent-glow: rgba(99, 102, 241, 0.4); }
        body { background: #020617; color: #f1f5f9; overflow-x: hidden; font-family: 'Space Grotesk', sans-serif; }
        
        /* Premium Glassmorphism */
        .glass { background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(25px); border: 1px solid rgba(255,255,255,0.08); }
        .glass-dark { background: rgba(0, 0, 0, 0.6); backdrop-filter: blur(15px); border: 1px solid rgba(255,255,255,0.1); }
        
        /* Balanced Grid */
        #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 2rem; padding-bottom: 5rem; }
        
        .browser-card { 
            background: #0f172a; border: 1px solid rgba(255,255,255,0.05); border-radius: 1.5rem;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative; overflow: hidden;
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        }
        .browser-card:hover { border-color: var(--accent); transform: translateY(-4px); box-shadow: 0 20px 50px -15px rgba(0,0,0,0.8); }
        .browser-card.selected { border: 2px solid var(--accent); box-shadow: 0 0 40px var(--accent-glow); }
        
        /* Interaction Surface */
        .surface-wrapper { position: relative; aspect-ratio: 16/9; background: #000; overflow: hidden; }
        canvas, video { width: 100%; height: 100%; object-fit: contain; cursor: crosshair; outline: none; }
        
        .debug-tag { font-family: 'JetBrains Mono', monospace; font-size: 8px; color: #64748b; }
        #debug-hud { position: fixed; bottom: 20px; right: 20px; z-index: 9999; background: rgba(0,0,0,0.8); padding: 12px; border-radius: 12px; font-size: 10px; pointer-events: none; max-height: 200px; overflow: hidden; border: 1px solid var(--accent); }
        
        .btn-brand { background: var(--accent); color: white; transition: 0.2s; }
        .btn-brand:hover { filter: brightness(1.2); transform: scale(1.05); }
        .btn-nav { width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.05); border-radius: 8px; transition: 0.2s; color: #94a3b8; }
        .btn-nav:hover { background: var(--accent); color: white; transform: rotate(5deg); }
    </style>
</head>
<body class="font-sans">
    <div id="debug-hud">System Status: Waiting...</div>

    <!-- Navbar -->
    <nav class="glass sticky top-0 z-[100] px-8 py-4 flex items-center justify-between">
        <div class="flex items-center gap-4">
            <div class="w-12 h-12 bg-gradient-to-br from-brand to-pink-500 rounded-2xl flex items-center justify-center shadow-lg shadow-brand/20">
                <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
            </div>
            <div>
                <h1 class="text-2xl font-black tracking-tighter uppercase italic">Stealth<span class="text-brand">Node</span></h1>
                <div class="flex items-center gap-2 text-[10px] text-slate-500 font-bold tracking-widest uppercase">
                    <span class="dot-ping w-1.5 h-1.5 bg-green-500 rounded-full"></span> SYSTEM ONLINE
                </div>
            </div>
        </div>

        <div class="flex items-center gap-6">
            <div class="hidden md:flex gap-8 text-xs font-mono">
                <div class="flex flex-col"><span class="text-slate-500 uppercase">Load Balance</span><span id="stat-nodes" class="text-white">0 Nodes Connected</span></div>
                <div class="flex flex-col"><span class="text-slate-500 uppercase">Pool Size</span><span id="stat-browsers" class="text-white">0 Active Sessions</span></div>
                <div class="flex flex-col"><span class="text-slate-500 uppercase">Sync Level</span><span id="stat-selected" class="text-brand font-bold">0 Synced</span></div>
            </div>
            
            <div class="flex items-center bg-black/40 rounded-2xl p-1 border border-white/5">
                <input id="batch-count" type="number" value="1" min="1" max="10" class="w-12 bg-transparent text-center text-sm font-bold outline-none">
                <button onclick="requestNewBrowser()" id="req-btn" class="bg-brand hover:brightness-110 text-white px-6 py-2.5 rounded-xl text-xs font-extrabold transition-all shadow-xl shadow-brand/20 active:scale-95">
                    SPAWN BATCH
                </button>
            </div>
        </div>
    </nav>

    <!-- Master Controls -->
    <div class="px-8 mt-8">
        <div class="glass p-6 rounded-[2.5rem] flex flex-wrap items-center justify-between gap-6 border-white/5 shadow-2xl">
            <div class="flex items-center gap-3">
                <button onclick="selectAll()" class="px-5 py-2.5 bg-white/5 hover:bg-white/10 rounded-2xl text-xs font-bold transition">SELECT ALL</button>
                <button onclick="deselectAll()" class="px-5 py-2.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-2xl text-xs font-bold transition">CLEAR</button>
                <div class="w-px h-8 bg-white/10 mx-2"></div>
                <label class="flex items-center gap-3 cursor-pointer group">
                    <div class="relative">
                        <input type="checkbox" id="follow-mode" class="sr-only peer" onchange="toggleFollow()">
                        <div class="w-10 h-6 bg-slate-700 rounded-full peer-checked:bg-brand transition"></div>
                        <div class="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition peer-checked:translate-x-4"></div>
                    </div>
                    <span class="text-xs font-bold text-slate-400 group-hover:text-white transition uppercase">Follow Mode</span>
                </label>
            </div>

            <div class="flex-1 max-w-2xl px-4 py-2 bg-black/40 rounded-3xl border border-white/5 flex items-center gap-4 focus-within:border-brand/40 transition">
                <svg class="w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" /></svg>
                <input id="mass-url" type="text" placeholder="Direct Command: Enter URL to all selected..." class="bg-transparent border-none outline-none flex-1 py-1 text-sm font-medium">
                <button onclick="massNavigate()" class="bg-brand text-white px-6 py-2 rounded-2xl text-[10px] font-black uppercase tracking-wider">EXECUTE</button>
            </div>

            <div class="flex items-center gap-3">
                <button onclick="massAction('refresh')" class="group p-3 bg-white/5 hover:bg-green-500/20 text-slate-400 hover:text-green-500 rounded-2xl transition" title="Refresh Selected">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
                </button>
                <button onclick="massAction('close')" class="group p-3 bg-white/5 hover:bg-red-500/20 text-slate-400 hover:text-red-500 rounded-2xl transition" title="Close Selected">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                </button>
            </div>
        </div>
    </div>

    <!-- Main View -->
    <main class="p-8">
        <div id="grid">
            <!-- Cards injected here -->
        </div>
    </main>

    <!-- Detailed Control Modal -->
    <div id="modal" class="fixed inset-0 z-[200] hidden bg-slate-950/95 backdrop-blur-2xl flex flex-col">
        <div class="p-6 border-b border-white/5 flex items-center justify-between">
            <div class="flex items-center gap-4">
                <div class="px-4 py-1.5 bg-brand text-white text-[10px] font-black rounded-full shadow-lg shadow-brand/20 uppercase tracking-widest">LIVE INTERACTION</div>
                <h3 id="modal-title" class="text-sm font-bold text-slate-400 tracking-tight">NODE: 001 / ID: FF99</h3>
            </div>
            <div class="flex items-center gap-4">
                <div class="flex items-center px-4 py-2 bg-black/40 border border-white/5 rounded-2xl gap-3">
                    <button onclick="modalNav('back')" class="text-slate-500 hover:text-white transition">←</button>
                    <button onclick="modalNav('forward')" class="text-slate-500 hover:text-white transition">→</button>
                    <div class="w-px h-4 bg-white/10 mx-1"></div>
                    <button onclick="modalNav('refresh')" class="text-slate-500 hover:text-white transition">↻</button>
                </div>
                <input id="modal-url" type="text" class="bg-black/40 border border-white/5 rounded-2xl px-6 py-2.5 text-xs w-[400px] focus:border-brand/40 outline-none transition" placeholder="Enter URL...">
                <button onclick="modalNavGo()" class="bg-brand text-white px-6 py-2.5 rounded-2xl text-[10px] font-black">NAVIGATE</button>
                <div class="w-px h-8 bg-white/10 mx-2"></div>
                <button onclick="closeModal()" class="w-10 h-10 flex items-center justify-center bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-white rounded-2xl transition">×</button>
            </div>
        </div>
        <div class="flex-1 relative bg-black flex items-center justify-center overflow-hidden p-6">
            <video id="modal-video" class="max-w-full max-h-full rounded-2xl shadow-2xl shadow-brand/10 border border-white/5 cursor-crosshair" autoplay playsinline muted></video>
            <canvas id="modal-canvas" class="hidden max-w-full max-h-full rounded-2xl shadow-2xl border border-white/5 cursor-crosshair"></canvas>
        </div>
    </div>

    <script>
        let nodes = {};
        let instances = []; // { key, nid, bid, url, ws, dc, pc, mode }
        let selectedKeys = new Set();
        let followMode = false;
        let activeModalKey = null;

        // Initialize Hub connection
        function initHub() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(`${protocol}//${location.host}/ws/hub`);
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'update') {
                    nodes = msg.nodes;
                    syncInstances();
                    updateStats();
                }
            };
            ws.onclose = () => setTimeout(initHub, 1000);
        }

        function updateStats() {
            let totalBrowsers = 0;
            Object.values(nodes).forEach(n => totalBrowsers += n.browsers.length);
            
            document.getElementById('stat-nodes').textContent = Object.keys(nodes).length;
            document.getElementById('stat-browsers').textContent = totalBrowsers;
            document.getElementById('stat-selected').textContent = selectedKeys.size;
        }

        function syncInstances() {
            const currentKeys = new Set();
            Object.entries(nodes).forEach(([nid, node]) => {
                node.browsers.forEach(bid => {
                    currentKeys.add(`${nid}::${bid}`);
                });
            });

            // Remove stale
            instances = instances.filter(inst => {
                if (!currentKeys.has(inst.key)) {
                    document.getElementById(`card-${inst.key.replace(/[:]/g, '-')}`)?.remove();
                    inst.ws.close();
                    inst.pc?.close();
                    selectedKeys.delete(inst.key);
                    return false;
                }
                return true;
            });

            // Add new
            Object.entries(nodes).forEach(([nid, node]) => {
                node.browsers.forEach(bid => {
                    const key = `${nid}::${bid}`;
                    if (!instances.find(i => i.key === key)) {
                        createInstance(nid, bid, node.url);
                    }
                });
            });
            
            renderGrid();
        }

        function createInstance(nid, bid, nodeUrl) {
            const key = `${nid}::${bid}`;
            const wsUrl = nodeUrl.replace('http', 'ws') + '/ws/' + bid;
            const ws = new WebSocket(wsUrl);
            
            const inst = { key, nid, bid, url: nodeUrl, ws, pc: null, dc: null, mode: 'ws', canvas: null, video: null };
            instances.push(inst);

            ws.onmessage = (e) => {
                const m = JSON.parse(e.data);
                if (m.type === 'frame') {
                    const img = new Image();
                    img.src = 'data:image/jpeg;base64,' + m.data;
                    img.onload = () => {
                        if (inst.mode === 'ws' && inst.canvas) {
                            inst.canvas.width = img.width;
                            inst.canvas.height = img.height;
                            const ctx = inst.canvas.getContext('2d');
                            ctx.drawImage(img, 0, 0);
                        }
                        if (activeModalKey === key && inst.mode === 'ws') {
                            const mv = document.getElementById('modal-canvas');
                            mv.width = img.width;
                            mv.height = img.height;
                            mv.getContext('2d').drawImage(img, 0, 0);
                        }
                    };
                }
            };

            setupWebRTC(inst);
        }

        async function setupWebRTC(inst) {
            try {
                const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
                inst.pc = pc;

                pc.ontrack = (e) => {
                    if (inst.video) {
                        inst.video.srcObject = e.streams[0];
                        inst.video.onloadedmetadata = () => {
                            inst.video.play();
                            inst.mode = 'webrtc';
                            updateGridItem(inst);
                        };
                    }
                };

                // Create DataChannel
                const dc = pc.createDataChannel("control", { ordered: false });
                inst.dc = dc;

                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);

                const res = await fetch(`${inst.url}/api/offer/${inst.bid}`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
                });
                
                const answer = await res.json();
                if (answer.error) {
                    debug(`RTC Error: ${answer.error}. Using WebSocket.`);
                    inst.mode = 'ws';
                    updateGridItem(inst);
                    if (activeModalKey === inst.key) syncModalSurface(inst);
                } else {
                    await pc.setRemoteDescription(new RTCSessionDescription(answer));
                }
            } catch (e) {
                console.error("RTC Failed for", inst.key, e);
            }
        }

        function renderGrid() {
            const grid = document.getElementById('grid');
            instances.forEach(inst => {
                const safeId = inst.key.replace(/[:]/g, '-');
                let card = document.getElementById(`card-${safeId}`);
                if (!card) {
                    card = document.createElement('div');
                    card.id = `card-${safeId}`;
                    card.className = "browser-card group";
                    card.innerHTML = `
                        <!-- Compact Header -->
                        <div class="px-4 py-3 flex items-center justify-between bg-black/20 border-b border-white/5">
                            <div class="flex items-center gap-3">
                                <input type="checkbox" onchange="toggleSelect('${inst.key}')" class="w-4 h-4 rounded border-white/10 bg-black/40 checked:bg-brand transition cursor-pointer">
                                <div>
                                    <div class="text-xs font-bold text-white flex items-center gap-2">
                                        ${inst.bid.substring(0,8)} 
                                        <span id="mode-tag-${safeId}" class="text-[8px] opacity-70">...</span>
                                    </div>
                                </div>
                            </div>
                            <div class="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition">
                                <button onclick="navAction('${inst.key}', 'back')" class="btn-nav">←</button>
                                <button onclick="navAction('${inst.key}', 'forward')" class="btn-nav">→</button>
                                <button onclick="navAction('${inst.key}', 'refresh')" class="btn-nav">↻</button>
                                <button onclick="closeBrowser('${inst.key}')" class="btn-nav hover:bg-red-500/20 hover:text-red-500">×</button>
                                <button onclick="openModal('${inst.key}')" class="btn-nav bg-brand/20 text-brand">⛶</button>
                            </div>
                        </div>
                        
                        <!-- Large Visual Surface -->
                        <div class="surface-wrapper">
                            <canvas id="view-ws-${safeId}"></canvas>
                            <video id="view-rtc-${safeId}" autoplay playsinline muted class="hidden"></video>
                            
                            <!-- Quick Nav Input (Hover Only) -->
                            <div class="absolute bottom-4 left-4 right-4 translate-y-2 opacity-0 group-hover:translate-y-0 group-hover:opacity-100 transition-all z-10">
                                <div class="bg-black/60 backdrop-blur-md rounded-xl p-1 flex items-center border border-white/10">
                                    <input id="url-${safeId}" type="text" placeholder="https://..." class="bg-transparent border-none outline-none flex-1 px-3 py-1.5 text-[11px] text-white">
                                    <button onclick="navSingle('${inst.key}')" class="bg-brand text-white px-3 py-1.5 rounded-lg text-[10px] font-bold">GO</button>
                                </div>
                            </div>
                        </div>
                    `;
                    grid.appendChild(card);
                    inst.canvas = document.getElementById(`view-ws-${safeId}`);
                    inst.video = document.getElementById(`view-rtc-${safeId}`);
                    
                    bindSurfaceEvents(inst.canvas, inst.key);
                    bindSurfaceEvents(inst.video, inst.key);
                }
                
                const cardEl = document.getElementById(`card-${safeId}`);
                const cb = cardEl.querySelector('input[type="checkbox"]');
                if (cb) cb.checked = selectedKeys.has(inst.key);
                cardEl.classList.toggle('selected', selectedKeys.has(inst.key));
                
                updateGridItem(inst);
            });
        }

        function updateGridItem(inst) {
            const safeId = inst.key.replace(/[:]/g, '-');
            const v_ws = document.getElementById(`view-ws-${safeId}`);
            const v_rtc = document.getElementById(`view-rtc-${safeId}`);
            const tag = document.getElementById(`mode-tag-${safeId}`);
            
            if (inst.mode === 'webrtc') {
                v_ws?.classList.add('hidden');
                v_rtc?.classList.remove('hidden');
                if (tag) { tag.textContent = 'WebRTC ⚡'; tag.style.background = 'rgba(34, 197, 94, 0.1)'; tag.style.color = '#22c55e'; }
            } else {
                v_rtc?.classList.add('hidden');
                v_ws?.classList.remove('hidden');
                if (tag) { tag.textContent = 'WebSocket 📡'; tag.style.background = 'rgba(99, 102, 241, 0.1)'; tag.style.color = '#6366f1'; }
            }
        }

        function toggleSelect(key) {
            if (selectedKeys.has(key)) selectedKeys.delete(key);
            else selectedKeys.add(key);
            renderGrid();
            updateStats();
        }

        function selectAll() {
            instances.forEach(i => selectedKeys.add(i.key));
            renderGrid();
            updateStats();
        }

        function deselectAll() {
            selectedKeys.clear();
            renderGrid();
            updateStats();
        }

        function toggleFollow() {
            followMode = document.getElementById('follow-mode').checked;
        }

        function debug(msg) {
            const hud = document.getElementById('debug-hud');
            const entry = document.createElement('div');
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
            hud.prepend(entry);
            if (hud.children.length > 10) hud.lastChild.remove();
        }

        function sendMsg(key, payload) {
            const inst = instances.find(i => i.key === key);
            if (!inst) return;
            
            debug(`Sending ${payload.type} to ${inst.bid}`);
            
            // Try DataChannel first if open for extreme speed
            if (inst.dc && inst.dc.readyState === 'open') {
                inst.dc.send(JSON.stringify(payload));
            } else if (inst.ws.readyState === WebSocket.OPEN) {
                inst.ws.send(JSON.stringify(payload));
            } else {
                debug(`Send failed: all channels closed for ${inst.bid}`);
            }
        }

        function massNavigate() {
            const url = document.getElementById('mass-url').value;
            if (!url) return;
            selectedKeys.forEach(k => sendMsg(k, { type: 'navigate', url }));
        }

        function navSingle(key) {
            const safeId = key.replace(/[:]/g, '-');
            const url = document.getElementById(`url-${safeId}`).value;
            if (url) sendMsg(key, { type: 'navigate', url });
        }

        function massAction(type) {
            if (type === 'close' && !confirm('Close selected browsers?')) return;
            selectedKeys.forEach(k => {
                if (type === 'close') {
                    const inst = instances.find(i => i.key === k);
                    if (inst) fetch(`${inst.url}/close/${inst.bid}`, { method: 'POST' });
                } else {
                    sendMsg(k, { type: 'refresh' });
                }
            });
        }

        // Modal Controls
        function openModal(key) {
            const inst = instances.find(i => i.key === key);
            if (!inst) return;
            activeModalKey = key;
            const modal = document.getElementById('modal');
            modal.classList.remove('hidden');
            document.getElementById('modal-title').textContent = `NODE: ${inst.nid} / SESSION: ${inst.bid}`;
            
            syncModalSurface(inst);
        }

        function syncModalSurface(inst) {
            const mv = document.getElementById('modal-video');
            const mc = document.getElementById('modal-canvas');
            
            if (inst.mode === 'webrtc') {
                mv.classList.remove('hidden');
                mc.classList.add('hidden');
                if (inst.video && inst.video.srcObject) {
                    mv.srcObject = inst.video.srcObject;
                    mv.play().catch(() => {});
                }
                bindSurfaceEvents(mv, inst.key);
            } else {
                mc.classList.remove('hidden');
                mv.classList.add('hidden');
                bindSurfaceEvents(mc, inst.key);
            }
        }

        function closeModal() {
            activeModalKey = null;
            document.getElementById('modal').classList.add('hidden');
            window.onkeydown = null;
            const mv = document.getElementById('modal-video');
            mv.srcObject = null;
        }

        function navAction(key, type) {
            sendMsg(key, { type });
        }

        async function closeBrowser(key) {
            const inst = instances.find(i => i.key === key);
            if (!inst) return;
            if (confirm(`Kill session ${inst.bid}?`)) {
                await fetch(`${inst.url}/close/${inst.bid}`, { method: 'POST' });
                selectedKeys.delete(key);
            }
        }

        let focusedKey = null;

        let isDragging = false;
        let lastMove = 0;

        function bindSurfaceEvents(el, key) {
            const getCoords = (e) => {
                const rect = el.getBoundingClientRect();
                const scaleX = 1280 / rect.width;
                const scaleY = 720 / rect.height;
                return {
                    x: Math.round((e.clientX - rect.left) * scaleX),
                    y: Math.round((e.clientY - rect.top) * scaleY)
                };
            };

            const handler = (type, e) => {
                const coords = getCoords(e);
                const payload = { 
                    type, 
                    x: coords.x, 
                    y: coords.y, 
                    deltaY: e.deltaY ? Math.round(e.deltaY * 1.5) : 0,
                    key: e.key,
                    button: e.button === 2 ? "right" : "left"
                };

                sendMsg(key, payload);
                if (followMode && selectedKeys.has(key)) {
                    selectedKeys.forEach(k => { if (k !== key) sendMsg(k, payload); });
                }
            };

            el.onmousedown = (e) => { 
                focusedKey = key; 
                isDragging = true;
                el.focus(); 
                handler('mousedown', e);
                
                // Track globally to handle dragging outside the element
                const onGlobalMove = (me) => {
                    if (!isDragging) return;
                    const now = performance.now();
                    if (now - lastMove < 25) return; // ~40fps throttle
                    lastMove = now;
                    handler('mousemove', me);
                };
                
                const onGlobalUp = (ue) => {
                    isDragging = false;
                    handler('mouseup', ue);
                    window.removeEventListener('mousemove', onGlobalMove);
                    window.removeEventListener('mouseup', onGlobalUp);
                };
                
                window.addEventListener('mousemove', onGlobalMove);
                window.addEventListener('mouseup', onGlobalUp);
            };

            el.onclick = (e) => { if (!isDragging) handler('click', e); };
            el.oncontextmenu = (e) => { e.preventDefault(); handler('click', { ...e, button: 2 }); };
            el.onwheel = (e) => { e.preventDefault(); handler('scroll', e); };
            el.tabIndex = 0; 
        }

        // Global Key Listener
        window.addEventListener('keydown', (e) => {
            if (document.activeElement.tagName === 'INPUT') return;
            if (!focusedKey && !activeModalKey) return;

            const key = activeModalKey || focusedKey;
            
            // Critical: Don't let space/arrows scroll the page while controlling
            const blocked = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Backspace', 'Tab', 'Enter', ' '];
            if (blocked.includes(e.key)) e.preventDefault();

            const payload = { type: 'key', key: e.key };
            sendMsg(key, payload);

            // Broadcast if needed
            if (followMode && selectedKeys.has(key)) {
                selectedKeys.forEach(k => { if (k !== key) sendMsg(k, payload); });
            }
        });

        function modalNav(type) {
            if (!activeModalKey) return;
            const targets = followMode ? Array.from(selectedKeys) : [activeModalKey];
            targets.forEach(k => sendMsg(k, { type }));
        }

        function modalNavGo() {
            if (!activeModalKey) return;
            const url = document.getElementById('modal-url').value;
            if (!url) return;
            const targets = followMode ? Array.from(selectedKeys) : [activeModalKey];
            targets.forEach(k => sendMsg(k, { type: 'navigate', url }));
        }

        async function requestNewBrowser() {
            const btn = document.getElementById('req-btn');
            const count = document.getElementById('batch-count').value;
            btn.disabled = true;
            btn.textContent = `SPAWNING ${count}...`;
            try {
                await fetch(`/api/request_browser?count=${count}`);
            } catch (e) {
                alert("Failed to allocate: " + e.message);
            }
            btn.disabled = false;
            btn.textContent = "SPAWN BATCH";
        }

        initHub();
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
