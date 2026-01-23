// ==========================================================
// StealthNode Command Center - Core Application
// ==========================================================

class StealthNodeApp {
    constructor() {
        // State
        this.nodes = {};
        this.cachedNodes = {}; // Cache for offline resilience
        this.instances = []; // { key, nid, bid, url, ws, dc, pc, mode, canvas, video }
        this.selectedKeys = new Set();
        this.followMode = false;
        this.activeModalKey = null;
        this.spawnMode = 'ephemeral';
        this.currentView = 'browsers';
        this.focusedKey = null;
        this.isDragging = false;
        this.lastMove = 0;
        this.isOffline = false;
        this.autoRefresh = true; // Auto-sync enabled by default

        // Connection state
        this.connectionState = 'disconnected';
        this.hubWs = null;
        this.connectionAttempts = 0;
        this.lastConnectionAttempt = 0;
        this.lastConnected = 0;
        this.lastMessageTime = 0;
        this.lastPongTime = 0;
        this.consecutiveErrors = 0;
        this.healthCheckInterval = null;

        // Bind methods
        this.init = this.init.bind(this);
        this.initHub = this.initHub.bind(this);
        this.fetchState = this.fetchState.bind(this); // New method
        this.syncInstances = this.syncInstances.bind(this);
        this.setOfflineMode = this.setOfflineMode.bind(this);
        this.setOnlineMode = this.setOnlineMode.bind(this);
        this.updateConnectionStatus = this.updateConnectionStatus.bind(this);
        this.startConnectionHealthCheck = this.startConnectionHealthCheck.bind(this);
    }

    // Initialize the application
    init() {
        this.fetchState(); // Initial sync
        this.initHub();
        this.bindGlobalEvents();
        this.showSkeletonLoading();

        // Polling fallback every 5s in case WS is blocked
        setInterval(() => {
            if (this.autoRefresh) this.fetchState();
        }, 5000);

        console.log('[StealthNode] Application initialized');
    }

    // Toggle auto-refresh behavior
    toggleAutoRefresh() {
        this.autoRefresh = !this.autoRefresh;
        const btn = document.getElementById('auto-refresh-btn');
        if (btn) {
            btn.classList.toggle('active', this.autoRefresh);
            btn.innerHTML = this.autoRefresh ?
                '<span class="icon">🔄</span> Auto Sync: ON' :
                '<span class="icon">⏸️</span> Auto Sync: OFF';
        }
        this.showToast(`Auto-sync ${this.autoRefresh ? 'enabled' : 'disabled'}`, 'info');

        if (this.autoRefresh) {
            this.fetchState(); // Immediate sync when turning ON
        }
    }

    async fetchState() {
        try {
            const res = await fetch('/api/nodes');
            if (res.status === 401) {
                window.location.href = '/login';
                return;
            }
            if (!res.ok) throw new Error('State fetch failed');

            const data = await res.json();

            // Handle auto-refresh logic
            if (!this.autoRefresh && this.instances.length > 0) {
                this.updateStats();
                return;
            }

            if (this.currentView === 'browsers') {
                // Ensure data is synchronized and the UI is rendered immediately
                this.nodes = data.nodes || {};
                this.syncInstances();
                this.renderGrid();
                this.updateStats();
            }
        } catch (e) {
            console.error('[StealthNode] Failed to fetch state:', e);
            if (this.connectionState !== 'connected') {
                this.setOfflineMode();
            }
        }
    }

    // Show skeleton loading cards
    showSkeletonLoading(count = 3) {
        const grid = document.getElementById('browser-grid');
        if (!grid) return;

        grid.innerHTML = '';
        for (let i = 0; i < count; i++) {
            const skeleton = document.createElement('div');
            skeleton.className = 'skeleton-card';
            skeleton.id = `skeleton-${i}`;
            skeleton.innerHTML = `
                <div class="skeleton-card-header">
                    <div class="flex items-center gap-md">
                        <div class="skeleton skeleton-checkbox"></div>
                        <div class="flex flex-col gap-xs">
                            <div class="skeleton skeleton-text" style="width: 100px;"></div>
                            <div class="skeleton skeleton-text-sm"></div>
                        </div>
                    </div>
                    <div class="flex gap-sm">
                        <div class="skeleton" style="width: 32px; height: 32px;"></div>
                        <div class="skeleton" style="width: 32px; height: 32px;"></div>
                        <div class="skeleton" style="width: 32px; height: 32px;"></div>
                    </div>
                </div>
                <div class="skeleton skeleton-surface"></div>
            `;
            grid.appendChild(skeleton);
        }
    }

    // Remove skeleton loading
    removeSkeletonLoading() {
        document.querySelectorAll('.skeleton-card').forEach(el => el.remove());
    }

    // Set offline mode - show cached data with offline indicators
    setOfflineMode() {
        if (this.isOffline) return;
        this.isOffline = true;
        console.log('[StealthNode] Entering offline mode');
        this.nodes = this.cachedNodes; // Use cached data
        this.syncInstances();
        this.updateStats();
        this.showOfflineNotification();
    }

    // Set online mode - resume normal operation
    setOnlineMode() {
        if (!this.isOffline) return;
        this.isOffline = false;
        console.log('[StealthNode] Entering online mode');
        this.hideOfflineNotification();
    }

    // Show offline notification
    showOfflineNotification() {
        let notification = document.getElementById('offline-notification');
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'offline-notification';
            notification.className = 'offline-notification';
            document.body.appendChild(notification);
        }
        notification.innerHTML = `
            <div class="flex items-center gap-md">
                <div class="flex items-center gap-sm">
                    <span class="icon" style="font-size: 1.2rem;">⚠️</span>
                    <div class="flex flex-col">
                        <span style="font-weight: 600; color: var(--color-text-primary);">Connection Lost</span>
                        <span style="font-size: 0.75rem; color: var(--color-text-muted);">Real-time updates paused. Reconnecting...</span>
                    </div>
                </div>
                <button onclick="app.initHub()" class="btn btn-primary btn-sm" style="padding: 4px 12px; font-size: 0.75rem;">
                    Reconnect Now
                </button>
            </div>
        `;
        notification.style.display = 'block';
    }

    // Hide offline notification
    hideOfflineNotification() {
        const notification = document.getElementById('offline-notification');
        if (notification) {
            notification.style.display = 'none';
        }
    }

    // Initialize WebSocket connection to hub with professional connection management
    initHub() {
        // Clean up any existing connection
        if (this.hubWs) {
            this.hubWs.close();
            this.hubWs = null;
        }

        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/ws/hub`;

        console.log(`[StealthNode] Connecting to hub: ${wsUrl}`);
        this.connectionState = 'connecting';
        this.updateConnectionStatus();

        const ws = new WebSocket(wsUrl);
        this.hubWs = ws;
        this.connectionAttempts = (this.connectionAttempts || 0) + 1;
        this.lastConnectionAttempt = Date.now();

        ws.onopen = () => {
            console.log('[StealthNode] Hub connection established');
            this.connectionState = 'connected';
            this.connectionAttempts = 0;
            this.lastConnected = Date.now();
            this.consecutiveErrors = 0;
            this.setOnlineMode();
            this.updateConnectionStatus();
            this.showToast('Connected to server', 'success');
        };

        ws.onmessage = (e) => {
            try {
                const msg = JSON.parse(e.data);

                if (msg.type === 'update') {
                    // Cache incoming nodes
                    this.cachedNodes = msg.nodes;

                    // Apply partial/diff update to avoid full UI re-render
                    this.applyPartialUpdate(msg.nodes);
                    this.lastMessageTime = Date.now();
                } else if (msg.type === 'pong') {
                    // Handle ping/pong for connection health
                    this.lastPongTime = Date.now();
                }
            } catch (err) {
                console.error('[StealthNode] Failed to parse message:', err);
            }
        };

        ws.onclose = (event) => {
            console.log(`[StealthNode] Hub connection closed (code: ${event.code}, reason: ${event.reason})`);
            this.connectionState = 'disconnected';
            this.hubWs = null;
            this.setOfflineMode();
            this.updateConnectionStatus();
            // Implement exponential backoff for reconnection with jitter
            // We still schedule the reconnect even if hidden, but also rely on visibilitychange
            if (document.hidden) {
                console.log('[StealthNode] Page hidden — reconnect will also trigger on visibilitychange');
            }

            const exp = Math.min(2 ** Math.min(this.connectionAttempts, 8), 64);
            const backoffDelay = Math.min(1000 * exp, 30000);
            const jitter = (Math.random() - 0.5) * 0.5 * backoffDelay; // ±25%
            const delay = Math.max(500, backoffDelay + jitter);

            console.log(`[StealthNode] Reconnecting in ${Math.round(delay)}ms (attempt ${this.connectionAttempts + 1})`);

            setTimeout(() => {
                if (this.connectionState !== 'connected') this.initHub();
            }, delay);
        };

        ws.onerror = (err) => {
            console.error('[StealthNode] Hub connection error:', err);
            this.consecutiveErrors = (this.consecutiveErrors || 0) + 1;
            this.connectionState = 'error';
            this.updateConnectionStatus();

            // Don't immediately enter offline mode on error - wait for close event
            if (this.consecutiveErrors > 3) {
                console.warn('[StealthNode] Multiple connection errors, entering offline mode');
                this.setOfflineMode();
            }
        };

        // Set up connection health monitoring
        this.startConnectionHealthCheck();
    }

    // Apply partial/diff update between existing nodes and newNodes
    applyPartialUpdate(newNodes) {
        if (!this.autoRefresh && this.instances.length > 0) {
            return;
        }
        // Force an immediate state adoption and UI render to ensure
        // all new browsers appear instantly as they are created.
        this.nodes = newNodes || {};
        this.syncInstances();
        this.renderGrid();
        this.updateStats();
    }

    removeInstanceByKey(key) {
        const idx = this.instances.findIndex(i => i.key === key);
        if (idx === -1) return;
        const inst = this.instances[idx];
        const safeId = inst.key.replace(/[:]/g, '-');
        document.getElementById(`card-${safeId}`)?.remove();
        try { inst.ws?.close(); } catch (e) { }
        try { inst.pc?.close(); } catch (e) { }
        this.selectedKeys.delete(inst.key);
        this.instances.splice(idx, 1);
    }

    // Connection health monitoring
    startConnectionHealthCheck() {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }

        this.healthCheckInterval = setInterval(() => {
            const now = Date.now();

            // Check if we're supposed to be connected but haven't received messages
            if (this.connectionState === 'connected') {
                if (this.lastMessageTime && now - this.lastMessageTime > 60000) { // 1 minute
                    console.warn('[StealthNode] No messages received for 60s, connection may be stale');
                    this.connectionState = 'stale';
                    this.updateConnectionStatus();
                }

                // Send periodic ping
                if (this.hubWs && this.hubWs.readyState === WebSocket.OPEN) {
                    try {
                        this.hubWs.send(JSON.stringify({ type: 'ping', timestamp: now }));
                    } catch (e) {
                        console.error('[StealthNode] Failed to send ping:', e);
                    }
                }
            }

            // Check for ping timeout
            if (this.lastPongTime && now - this.lastPongTime > 10000) { // 10 seconds
                console.warn('[StealthNode] Ping timeout, connection unhealthy');
                if (this.hubWs) {
                    this.hubWs.close();
                }
            }
        }, 10000); // Check every 10 seconds
    }

    // Update connection status indicator
    updateConnectionStatus() {
        let indicator = document.getElementById('connection-status');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'connection-status';
            indicator.className = 'connection-status';
            document.body.appendChild(indicator);
        }

        const statusInfo = {
            connecting: { text: 'Connecting...', class: 'connecting', icon: '🔄' },
            connected: { text: 'Connected', class: 'connected', icon: '🟢' },
            disconnected: { text: 'Disconnected', class: 'disconnected', icon: '🔴' },
            error: { text: 'Connection Error', class: 'error', icon: '❌' },
            stale: { text: 'Connection Stale', class: 'stale', icon: '🟡' }
        };

        const info = statusInfo[this.connectionState] || statusInfo.disconnected;
        indicator.className = `connection-status ${info.class}`;
        indicator.innerHTML = `${info.icon} ${info.text}`;

        // Add click handler for manual reconnection
        indicator.onclick = () => {
            if (this.connectionState !== 'connecting') {
                console.log('[StealthNode] Manual reconnection requested');
                this.initHub();
            }
        };
    }

    // Update statistics display
    updateStats() {
        let totalBrowsers = 0;
        Object.values(this.nodes).forEach(n => {
            // Safely count browsers, handling both array and undefined cases
            if (n.browsers && Array.isArray(n.browsers)) {
                totalBrowsers += n.browsers.length;
            }
        });

        const nodesEl = document.getElementById('stat-nodes');
        const browsersEl = document.getElementById('stat-browsers');
        const selectedEl = document.getElementById('stat-selected');

        if (nodesEl) nodesEl.textContent = Object.keys(this.nodes).length;
        if (browsersEl) browsersEl.textContent = totalBrowsers;
        if (selectedEl) selectedEl.textContent = this.selectedKeys.size;

        // Update sidebar badge
        const badge = document.getElementById('browsers-badge');
        if (badge) {
            badge.textContent = totalBrowsers;
            badge.className = this.isOffline ? 'nav-badge offline' : 'nav-badge';
        }

        // Add offline class to stats containers
        const statsContainer = document.querySelector('.stats-grid');
        if (statsContainer) statsContainer.classList.toggle('offline-mode', this.isOffline);
    }

    // Sync browser instances with nodes
    syncInstances() {
        const currentKeys = new Set();

        // Build set of current browser keys - properly extract ID from object or string
        Object.entries(this.nodes).forEach(([nid, node]) => {
            if (!node.browsers || !Array.isArray(node.browsers)) return;
            node.browsers.forEach(bidInfo => {
                // bidInfo might be a string (id) or object {id, profile_id, mode, ...}
                const bid = typeof bidInfo === 'object' && bidInfo !== null ? bidInfo.id : bidInfo;
                if (bid) {
                    currentKeys.add(`${nid}::${bid}`);
                }
            });
        });

        // Handle instances based on online/offline state
        if (this.isOffline) {
            // In offline mode, don't remove instances - just mark them as offline
            this.instances.forEach(inst => {
                this.updateInstanceOfflineStatus(inst, !currentKeys.has(inst.key));
            });
        } else {
            // In online mode, remove truly stale instances (not in currentKeys)
            this.instances = this.instances.filter(inst => {
                if (!currentKeys.has(inst.key)) {
                    const safeId = inst.key.replace(/[:]/g, '-');
                    document.getElementById(`card-${safeId}`)?.remove();
                    inst.ws?.close();
                    inst.pc?.close();
                    this.selectedKeys.delete(inst.key);
                    return false;
                }
                // Mark as online
                this.updateInstanceOfflineStatus(inst, false);
                return true;
            });
        }

        // Add new instances
        Object.entries(this.nodes).forEach(([nid, node]) => {
            if (!node.browsers || !Array.isArray(node.browsers)) return;
            node.browsers.forEach(bidInfo => {
                // bidInfo might be a string (id) or object {id, profile_id, mode, ...}
                const bid = typeof bidInfo === 'object' && bidInfo !== null ? bidInfo.id : bidInfo;
                if (!bid) return; // Skip invalid entries

                const mode = typeof bidInfo === 'object' && bidInfo !== null ? bidInfo.mode : 'ephemeral';
                const profileId = typeof bidInfo === 'object' && bidInfo !== null ? bidInfo.profile_id : null;
                const key = `${nid}::${bid}`;

                let inst = this.instances.find(i => i.key === key);
                if (!inst) {
                    this.createInstance(nid, bid, node.url, profileId, mode);
                    // Mark as offline if we're currently offline
                    if (this.isOffline) {
                        inst = this.instances.find(i => i.key === key);
                        if (inst) this.updateInstanceOfflineStatus(inst, true);
                    }
                } else {
                    // Update metadata if it changed
                    if (profileId) inst.profileId = profileId;
                    if (mode) inst.modeAttr = mode;
                }
            });
        });
    }

    // Update instance offline status
    updateInstanceOfflineStatus(inst, isOffline) {
        const safeId = inst.key.replace(/[:]/g, '-');
        const card = document.getElementById(`card-${safeId}`);
        if (card) {
            if (isOffline) {
                card.classList.add('offline');
                // Add offline indicator
                let indicator = card.querySelector('.offline-indicator');
                if (!indicator) {
                    indicator = document.createElement('div');
                    indicator.className = 'offline-indicator';
                    indicator.innerHTML = '<span class="icon">🔴</span> Offline';
                    card.querySelector('.card-header').appendChild(indicator);
                }
            } else {
                card.classList.remove('offline');
                const indicator = card.querySelector('.offline-indicator');
                if (indicator) indicator.remove();
            }
        }
    }

    // Helper to get a unique hash of the current browser setup
    getBrowserListHash() {
        const keys = [];
        Object.values(this.nodes).forEach(node => {
            if (!node.browsers || !Array.isArray(node.browsers)) return;
            node.browsers.forEach(b => {
                const id = typeof b === 'object' && b !== null ? b.id : b;
                if (id) keys.push(id);
            });
        });
        return keys.sort().join(',');
    }

    // Create a new browser instance
    createInstance(nid, bid, nodeUrl, profileId = null, modeAttr = 'ephemeral') {
        const key = `${nid}::${bid}`;

        // RELAY LOGIC: Use server tunnel if hosted on HTTPS or node is remote
        // This bypasses mix-content blocking and NAT issues.
        const isHttps = window.location.protocol === 'https:';
        const nodeHost = new URL(nodeUrl).hostname;
        const currentHost = window.location.hostname;

        // Force relay if we are on HTTPS (due to mixed content) or if it's a remote node
        const shouldUseRelay = isHttps || (nodeHost !== 'localhost' && nodeHost !== '127.0.0.1' && nodeHost !== currentHost);

        let finalWsUrl;
        if (shouldUseRelay) {
            const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            finalWsUrl = `${proto}//${window.location.host}/ws/ui/stream/${nid}/${bid}`;
            console.log(`[NETWORK] Routing browser ${bid} through Server Relay (${finalWsUrl})`);
        } else {
            finalWsUrl = nodeUrl.replace('http', 'ws') + '/ws/' + bid;
            console.log(`[NETWORK] Connecting directly to browser ${bid} (${finalWsUrl})`);
        }

        const ws = new WebSocket(finalWsUrl);

        const inst = {
            key,
            nid,
            bid,
            url: nodeUrl,
            profileId,
            modeAttr,
            ws,
            pc: null,
            dc: null,
            mode: 'ws', // connection mode (ws/rtc)
            lastUrl: 'about:blank',
            canvas: null,
            video: null
        };

        this.instances.push(inst);
        this.renderGrid();

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
                    if (this.activeModalKey === key && inst.mode === 'ws') {
                        const mc = document.getElementById('modal-canvas');
                        if (mc) {
                            mc.width = img.width;
                            mc.height = img.height;
                            mc.getContext('2d').drawImage(img, 0, 0);
                        }
                    }
                };
            }
        };

        this.setupWebRTC(inst);
    }

    // Setup WebRTC connection for an instance
    async setupWebRTC(inst) {
        try {
            const pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });
            inst.pc = pc;

            pc.ontrack = (e) => {
                if (inst.video) {
                    inst.video.srcObject = e.streams[0];
                    inst.video.onloadedmetadata = () => {
                        inst.video.play();
                        inst.mode = 'webrtc';
                        this.updateGridItem(inst);
                    };
                }
            };

            // Create DataChannel for low-latency control
            const dc = pc.createDataChannel("control", { ordered: false });
            inst.dc = dc;

            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            const res = await fetch(`${inst.url}/api/offer/${inst.bid}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type
                })
            });

            const answer = await res.json();
            if (answer.error) {
                console.log(`[RTC] Error for ${inst.bid}: ${answer.error}, using WebSocket`);
                inst.mode = 'ws';
                this.updateGridItem(inst);
            } else {
                await pc.setRemoteDescription(new RTCSessionDescription(answer));
            }
        } catch (e) {
            console.error('[RTC] Setup failed for', inst.key, e);
        }
    }

    // Render the browser grid
    renderGrid() {
        const grid = document.getElementById('browser-grid');
        if (!grid) return;

        // Remove skeleton loading
        this.removeSkeletonLoading();

        // Clear empty state if we now have instances
        if (this.instances.length > 0) {
            const emptyState = grid.querySelector('.empty-state');
            if (emptyState) grid.innerHTML = '';
        }

        // Show empty state if no instances
        if (this.instances.length === 0) {
            grid.innerHTML = `
                <div class="empty-state" style="grid-column: 1 / -1;">
                    <div class="empty-state-icon">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                    </div>
                    <h3 class="empty-state-title">No Active Browser Sessions</h3>
                    <p class="empty-state-description">
                        Spawn new browser instances using the controls above. 
                        Browsers will appear here once they're ready.
                    </p>
                    <button onclick="app.requestNewBrowser()" class="btn btn-primary btn-lg">
                        <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
                        </svg>
                        Spawn Browser
                    </button>
                </div>
            `;
            return;
        }

        this.instances.forEach(inst => {
            const safeId = inst.key.replace(/[:]/g, '-');
            let card = document.getElementById(`card-${safeId}`);

            if (!card) {
                card = document.createElement('div');
                card.id = `card-${safeId}`;
                card.className = 'browser-card';
                card.innerHTML = `
                    <div class="browser-card-header">
                        <div class="browser-card-info">
                            <input type="checkbox" 
                                   class="browser-checkbox" 
                                   onchange="app.toggleSelect('${inst.key}')"
                                   ${this.selectedKeys.has(inst.key) ? 'checked' : ''}>
                            <div class="browser-meta">
                                <div class="flex items-center gap-sm">
                                    <div class="browser-id">${inst.bid.substring(0, 8)}</div>
                                    <div id="browser-url-display-${safeId}" class="browser-current-url">about:blank</div>
                                </div>
                                <div class="flex gap-xs mt-xs">
                                    <span id="mode-tag-${safeId}" class="browser-mode-tag websocket">WebSocket</span>
                                    ${inst.modeAttr === 'persistent' ?
                        `<span class="browser-mode-tag persistent">Storage</span>` :
                        `<span class="browser-mode-tag ephemeral">No Storage</span>`}
                                </div>
                            </div>
                        </div>
                        <div class="browser-card-controls visible">
                            <button onclick="app.navAction('${inst.key}', 'back')" class="btn btn-icon btn-ghost btn-xs" title="Back">
                                <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                                </svg>
                            </button>
                            <button onclick="app.navAction('${inst.key}', 'forward')" class="btn btn-icon btn-ghost btn-xs" title="Forward">
                                <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
                                </svg>
                            </button>
                            <button onclick="app.navAction('${inst.key}', 'refresh')" class="btn btn-icon btn-ghost btn-xs" title="Refresh">
                                <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                </svg>
                            </button>
                            <button onclick="app.closeBrowser('${inst.key}')" class="btn btn-icon btn-danger btn-xs" title="Close">
                                <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                            <button onclick="app.openModal('${inst.key}')" class="btn btn-icon btn-primary btn-xs" title="Expand">
                                <svg width="14" height="14" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                        d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                                </svg>
                            </button>
                        </div>
                    </div>
                    
                    <div class="browser-surface">
                        <canvas id="view-ws-${safeId}"></canvas>
                        <video id="view-rtc-${safeId}" autoplay playsinline muted class="hidden"></video>
                        
                        <div class="browser-url-bar">
                            <input id="url-${safeId}" type="text" class="browser-url-input" 
                                   placeholder="Enter URL and press Enter...">
                            <button onclick="app.navSingle('${inst.key}')" class="btn btn-primary btn-sm">Go</button>
                        </div>
                    </div>
                `;
                grid.appendChild(card);

                inst.canvas = document.getElementById(`view-ws-${safeId}`);
                inst.video = document.getElementById(`view-rtc-${safeId}`);

                this.bindSurfaceEvents(inst.canvas, inst.key);
                this.bindSurfaceEvents(inst.video, inst.key);

                // Bind enter key for URL input
                const urlInput = document.getElementById(`url-${safeId}`);
                if (urlInput) {
                    urlInput.addEventListener('keypress', (e) => {
                        if (e.key === 'Enter') {
                            this.navSingle(inst.key);
                        }
                    });
                }
            }

            // Update selection state
            const cardEl = document.getElementById(`card-${safeId}`);
            const cb = cardEl?.querySelector('.browser-checkbox');
            if (cb) cb.checked = this.selectedKeys.has(inst.key);
            cardEl?.classList.toggle('selected', this.selectedKeys.has(inst.key));

            this.updateGridItem(inst);
        });
    }

    // Update a single grid item
    updateGridItem(inst) {
        const safeId = inst.key.replace(/[:]/g, '-');
        const wsView = document.getElementById(`view-ws-${safeId}`);
        const rtcView = document.getElementById(`view-rtc-${safeId}`);
        const tag = document.getElementById(`mode-tag-${safeId}`);
        const urlDisplay = document.getElementById(`browser-url-display-${safeId}`);

        if (urlDisplay && inst.lastUrl) {
            urlDisplay.textContent = inst.lastUrl === 'about:blank' ? 'New Tab' : inst.lastUrl;
            urlDisplay.title = inst.lastUrl;
        }

        if (inst.mode === 'webrtc') {
            wsView?.classList.add('hidden');
            rtcView?.classList.remove('hidden');
            if (tag) {
                tag.textContent = 'WebRTC ⚡';
                tag.className = 'browser-mode-tag webrtc';
            }
        } else {
            rtcView?.classList.add('hidden');
            wsView?.classList.remove('hidden');
            if (tag) {
                tag.textContent = 'WebSocket';
                tag.className = 'browser-mode-tag websocket';
            }
        }
    }

    // Bind mouse/keyboard events to a surface element
    bindSurfaceEvents(el, key) {
        if (!el) return;

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
                deltaY: e.deltaY || 0,
                key: e.key,
                button: e.button === 2 ? "right" : "left"
            };

            this.sendMsg(key, payload);
            if (this.followMode && this.selectedKeys.has(key)) {
                this.selectedKeys.forEach(k => {
                    if (k !== key) this.sendMsg(k, payload);
                });
            }
        };

        let startX, startY;

        el.onmousedown = (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.focusedKey = key;
            this.isDragging = false;
            const coords = getCoords(e);
            startX = coords.x;
            startY = coords.y;

            el.focus();
            handler('mousedown', e);

            const onGlobalMove = (me) => {
                if (Math.abs(me.movementX) > 0.1 || Math.abs(me.movementY) > 0.1) {
                    this.isDragging = true;
                }
                const now = performance.now();
                if (now - this.lastMove < 16) return;
                this.lastMove = now;
                handler('mousemove', me);
            };

            const onGlobalUp = (ue) => {
                ue.preventDefault();
                ue.stopPropagation();
                const coords = getCoords(ue);
                const dist = Math.sqrt(Math.pow(coords.x - startX, 2) + Math.pow(coords.y - startY, 2));

                if (!this.isDragging && dist < 5) {
                    handler('click', ue);
                } else {
                    handler('mouseup', ue);
                }

                this.isDragging = false;
                window.removeEventListener('mousemove', onGlobalMove);
                window.removeEventListener('mouseup', onGlobalUp);
            };

            window.addEventListener('mousemove', onGlobalMove);
            window.addEventListener('mouseup', onGlobalUp);
        };

        el.oncontextmenu = (e) => { e.preventDefault(); };
        el.onwheel = (e) => { e.preventDefault(); handler('scroll', e); };
        el.tabIndex = 0;
    }

    // Bind global keyboard events
    bindGlobalEvents() {
        window.addEventListener('keydown', (e) => {
            if (document.activeElement.tagName === 'INPUT') return;
            if (!this.focusedKey && !this.activeModalKey) return;

            const key = this.activeModalKey || this.focusedKey;

            const blocked = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Backspace', 'Tab', 'Enter', ' '];
            if (blocked.includes(e.key)) e.preventDefault();

            const payload = { type: 'key', key: e.key };
            this.sendMsg(key, payload);

            if (this.followMode && this.selectedKeys.has(key)) {
                this.selectedKeys.forEach(k => {
                    if (k !== key) this.sendMsg(k, payload);
                });
            }
        });

        // Reconnect when tab becomes active
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && (this.connectionState === 'disconnected' || this.connectionState === 'error')) {
                console.log('[StealthNode] Tab visible and disconnected — initiating reconnection');
                this.initHub();
            }
        });
    }

    // Send message to a browser instance
    sendMsg(key, payload) {
        const inst = this.instances.find(i => i.key === key);
        if (!inst) return;

        // Try DataChannel first for lower latency
        if (inst.dc && inst.dc.readyState === 'open') {
            inst.dc.send(JSON.stringify(payload));
        } else if (inst.ws && inst.ws.readyState === WebSocket.OPEN) {
            inst.ws.send(JSON.stringify(payload));
        }
    }

    // Toggle selection of a browser
    toggleSelect(key) {
        if (this.selectedKeys.has(key)) {
            this.selectedKeys.delete(key);
        } else {
            this.selectedKeys.add(key);
        }
        this.renderGrid();
        this.updateStats();
    }

    // Select all browsers
    selectAll() {
        this.instances.forEach(i => this.selectedKeys.add(i.key));
        this.renderGrid();
        this.updateStats();
    }

    // Deselect all browsers
    deselectAll() {
        this.selectedKeys.clear();
        this.renderGrid();
        this.updateStats();
    }

    // Toggle follow mode
    toggleFollow() {
        const toggle = document.getElementById('follow-mode');
        this.followMode = toggle?.checked || false;
    }

    // Set spawn mode
    setMode(mode) {
        this.spawnMode = mode;

        const ephBtn = document.getElementById('mode-ephemeral');
        const perBtn = document.getElementById('mode-persistent');
        const profileInput = document.getElementById('profile-id');

        if (ephBtn) ephBtn.classList.toggle('active', mode === 'ephemeral');
        if (perBtn) perBtn.classList.toggle('active', mode === 'persistent');
        if (profileInput) profileInput.classList.toggle('hidden', mode !== 'persistent');
    }

    // Navigate a single browser
    navSingle(key) {
        const safeId = key.replace(/[:]/g, '-');
        const urlInput = document.getElementById(`url-${safeId}`);
        const url = urlInput?.value;
        if (url) {
            this.sendMsg(key, { type: 'navigate', url });
            if (urlInput) urlInput.value = '';
        }
    }

    // Navigate action (back, forward, refresh)
    navAction(key, type) {
        this.sendMsg(key, { type });
    }

    // Mass navigate selected browsers
    massNavigate() {
        const urlInput = document.getElementById('mass-url');
        const url = urlInput?.value;
        if (!url) return;

        this.selectedKeys.forEach(k => this.sendMsg(k, { type: 'navigate', url }));
        if (urlInput) urlInput.value = '';
        this.showToast('Navigation sent to selected browsers', 'success');
    }

    // Mass action on selected browsers
    massAction(type) {
        if (type === 'close' && !confirm('Close selected browsers?')) return;

        this.selectedKeys.forEach(k => {
            if (type === 'close') {
                const inst = this.instances.find(i => i.key === k);
                if (inst) fetch(`/api/close_browser/${inst.nid}/${inst.bid}`, { method: 'POST' });
            } else if (type === 'back' || type === 'forward' || type === 'refresh') {
                this.sendMsg(k, { type });
            }
        });

        if (type === 'close') {
            this.selectedKeys.clear();
            this.showToast('Browsers closed', 'info');
        } else {
            this.showToast(`Action '${type}' sent to selected`, 'success');
        }
    }

    // Close a single browser
    async closeBrowser(key) {
        const inst = this.instances.find(i => i.key === key);
        if (!inst) return;

        if (confirm(`Close browser ${inst.bid.substring(0, 8)}?`)) {
            await fetch(`${inst.url}/close/${inst.bid}`, { method: 'POST' });
            this.selectedKeys.delete(key);
        }
    }

    // Request new browser(s) - Optimized for scale
    async requestNewBrowser() {
        const btn = document.getElementById('spawn-btn');
        const countInput = document.getElementById('batch-count');
        const profileInput = document.getElementById('profile-id');

        const count = parseInt(countInput?.value) || 1;
        const profileId = profileInput?.value || '';

        if (btn) {
            btn.disabled = true;
            btn.innerHTML = `
                <svg class="animate-spin" width="16" height="16" fill="none" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" opacity="0.25"></circle>
                    <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                </svg>
                ${count > 50 ? 'Processing...' : `Spawning ${count}...`}
            `;
        }

        // Show skeleton loading for expected new instances (max 20 to avoid UI overload)
        const skeletonCount = Math.min(count, 20);
        this.showSkeletonLoading(skeletonCount);

        try {
            let url = `/api/request_browser?count=${count}&mode=${this.spawnMode}`;
            if (profileId) url += `&profile_id=${encodeURIComponent(profileId)}`;

            const res = await fetch(url);

            if (!res.ok) {
                const error = await res.json();
                this.showToast(error.detail || 'Failed to spawn browsers', 'error');
                return;
            }

            const data = await res.json();

            // Build detailed feedback message
            let message = `Spawned ${data.count} browser(s)`;
            if (data.failed > 0) {
                message += ` (${data.failed} failed)`;
            }
            if (data.remaining_limit !== -1 && data.remaining_limit !== undefined) {
                message += ` • ${data.remaining_limit} remaining`;
            }

            const toastType = data.count === data.requested ? 'success' :
                data.count > 0 ? 'info' : 'error';
            this.showToast(message, toastType);

            // Immediately fetch state after a successful request to sync UI
            setTimeout(() => this.fetchState(), 1000);

        } catch (e) {
            this.showToast(`Failed to spawn: ${e.message}`, 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = `
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
                    </svg>
                    Spawn
                `;
            }
        }
    }

    // Open modal for focused interaction
    openModal(key) {
        const inst = this.instances.find(i => i.key === key);
        if (!inst) return;

        this.activeModalKey = key;
        const modal = document.getElementById('modal');
        const title = document.getElementById('modal-title');

        if (modal) modal.classList.add('active');
        if (title) title.textContent = `NODE: ${inst.nid} / SESSION: ${inst.bid}`;

        this.syncModalSurface(inst);
    }

    // Sync modal surface with instance
    syncModalSurface(inst) {
        const videoEl = document.getElementById('modal-video');
        const canvasEl = document.getElementById('modal-canvas');

        if (inst.mode === 'webrtc') {
            videoEl?.classList.remove('hidden');
            canvasEl?.classList.add('hidden');
            if (inst.video && inst.video.srcObject && videoEl) {
                videoEl.srcObject = inst.video.srcObject;
                videoEl.play().catch(() => { });
            }
            if (videoEl) this.bindSurfaceEvents(videoEl, inst.key);
        } else {
            canvasEl?.classList.remove('hidden');
            videoEl?.classList.add('hidden');
            if (canvasEl) this.bindSurfaceEvents(canvasEl, inst.key);
        }
    }

    // Close modal
    closeModal() {
        this.activeModalKey = null;
        const modal = document.getElementById('modal');
        const videoEl = document.getElementById('modal-video');

        if (modal) modal.classList.remove('active');
        if (videoEl) videoEl.srcObject = null;
    }

    // Modal navigation
    modalNav(type) {
        if (!this.activeModalKey) return;
        const targets = this.followMode ? Array.from(this.selectedKeys) : [this.activeModalKey];
        targets.forEach(k => this.sendMsg(k, { type }));
    }

    // Modal navigate to URL
    modalNavGo() {
        if (!this.activeModalKey) return;
        const urlInput = document.getElementById('modal-url');
        const url = urlInput?.value;
        if (!url) return;

        const targets = this.followMode ? Array.from(this.selectedKeys) : [this.activeModalKey];
        targets.forEach(k => this.sendMsg(k, { type: 'navigate', url }));
        if (urlInput) urlInput.value = '';
    }

    // Show toast notification
    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        const icons = {
            success: '<svg class="toast-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>',
            error: '<svg class="toast-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>',
            info: '<svg class="toast-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>'
        };

        toast.innerHTML = `
            ${icons[type] || icons.info}
            <span class="toast-message">${message}</span>
        `;

        container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'toast-in 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // Switch view (for navigation)
    switchView(view) {
        this.currentView = view;

        // Update nav items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.view === view);
        });

        // Navigate to view
        if (view === 'vault') {
            window.location.href = '/vault';
        } else if (view === 'browsers') {
            window.location.href = '/';
        }
    }
}

// Initialize the application
const app = new StealthNodeApp();
document.addEventListener('DOMContentLoaded', () => app.init());

// CSS animation for spinning
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .animate-spin {
        animation: spin 1s linear infinite;
    }
`;
document.head.appendChild(style);
