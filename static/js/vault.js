// ==========================================================
// StealthNode - Profile Vault Module
// ==========================================================

class ProfileVault {
    constructor() {
        this.profiles = {};
        this.loading = false;
    }

    // Initialize the vault page
    init() {
        this.showSkeletonLoading();
        this.loadProfiles();
    }

    // Show skeleton loading
    showSkeletonLoading(count = 6) {
        const grid = document.getElementById('vault-grid');
        if (!grid) return;

        grid.innerHTML = '';
        for (let i = 0; i < count; i++) {
            const skeleton = document.createElement('div');
            skeleton.className = 'profile-card skeleton-loading';
            skeleton.innerHTML = `
                <div class="profile-card-header">
                    <div class="skeleton" style="width: 48px; height: 48px; border-radius: 12px;"></div>
                    <div class="skeleton" style="width: 80px; height: 20px; border-radius: 20px;"></div>
                </div>
                <div class="skeleton" style="width: 70%; height: 16px; margin: 16px 0 8px; border-radius: 4px;"></div>
                <div class="skeleton" style="width: 40%; height: 12px; margin-bottom: 24px; border-radius: 4px;"></div>
                <div class="flex gap-sm">
                    <div class="skeleton" style="flex: 1; height: 36px; border-radius: 8px;"></div>
                    <div class="skeleton" style="width: 36px; height: 36px; border-radius: 8px;"></div>
                </div>
            `;
            grid.appendChild(skeleton);
        }
    }

    // Load profiles from all nodes
    async loadProfiles() {
        this.loading = true;

        try {
            const res = await fetch('/api/profiles');
            if (!res.ok) throw new Error('Failed to fetch profiles');

            this.profiles = await res.json();
            this.renderProfiles();
        } catch (e) {
            console.error('Failed to load profiles:', e);
            this.showError(e.message);
        }

        this.loading = false;
    }

    // Render profile cards
    renderProfiles() {
        const grid = document.getElementById('vault-grid');
        if (!grid) return;

        grid.innerHTML = '';

        let totalProfiles = 0;

        Object.entries(this.profiles).forEach(([nid, profiles]) => {
            profiles.forEach(profile => {
                totalProfiles++;
                const card = this.createProfileCard(nid, profile);
                grid.appendChild(card);
            });
        });

        // Update count badge
        const badge = document.getElementById('vault-count');
        if (badge) badge.textContent = totalProfiles;

        // Show empty state if no profiles
        if (totalProfiles === 0) {
            grid.innerHTML = `
                <div class="empty-state" style="grid-column: 1 / -1;">
                    <div class="empty-state-icon">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9l-2-2H5a2 2 0 01-2 2v10a2 2 0 012 2z" />
                        </svg>
                    </div>
                    <h3 class="empty-state-title">No Persistent Profiles Found</h3>
                    <p class="empty-state-description">
                        Persistent browser profiles will appear here once you create them.
                        Use the "Persistent" mode when spawning browsers to save session data.
                    </p>
                    <a href="/" class="btn btn-primary btn-lg">
                        <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
                        </svg>
                        Create New Session
                    </a>
                </div>
            `;
        }
    }

    // Create a profile card element
    createProfileCard(nodeId, profile) {
        const card = document.createElement('div');
        card.className = 'profile-card';

        const sizeDisplay = (profile.size / 1024 / 1024).toFixed(2);

        card.innerHTML = `
            <div class="profile-card-header">
                <div class="profile-icon">
                    <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                            d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9l-2-2H5a2 2 0 01-2 2v10a2 2 0 012 2z" />
                    </svg>
                </div>
                <div class="profile-node-tag">NODE: ${nodeId}</div>
            </div>
            <h3 class="profile-name">${profile.profile_id}</h3>
            <p class="profile-meta">${sizeDisplay} MB Storage</p>
            <div class="profile-actions">
                ${profile.is_active ? `
                    <div class="profile-status active">Session Active</div>
                ` : `
                    <button onclick="vault.launchProfile('${nodeId}', '${profile.profile_id}')" 
                            class="btn btn-primary" style="flex: 1;">
                        <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        Launch
                    </button>
                `}
                <button onclick="vault.deleteProfile('${nodeId}', '${profile.profile_id}')" 
                        class="btn btn-icon btn-danger" title="Delete Profile">
                    <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                </button>
            </div>
        `;

        return card;
    }

    // Launch a profile
    async launchProfile(nodeId, profileId) {
        try {
            this.showToast(`Launching session: ${profileId}`, 'info');

            const url = `/api/request_browser?count=1&mode=persistent&profile_id=${encodeURIComponent(profileId)}`;
            await fetch(url);

            // Redirect to command center
            window.location.href = '/';
        } catch (e) {
            this.showToast(`Launch failed: ${e.message}`, 'error');
        }
    }

    // Delete a profile
    async deleteProfile(nodeId, profileId) {
        if (!confirm(`Permanently delete profile "${profileId}"?\n\nThis will remove all stored cookies, sessions, and browser data. This action cannot be undone.`)) {
            return;
        }

        try {
            const res = await fetch(`/api/delete_profile/${nodeId}/${encodeURIComponent(profileId)}`, {
                method: 'DELETE'
            });

            const data = await res.json();

            if (data.status === 'ok') {
                this.showToast('Profile deleted successfully', 'success');
                this.loadProfiles();
            } else {
                throw new Error(data.message || 'Delete failed');
            }
        } catch (e) {
            this.showToast(`Delete failed: ${e.message}`, 'error');
        }
    }

    // Refresh vault
    refresh() {
        this.showSkeletonLoading();
        this.loadProfiles();
    }

    // Show error state
    showError(message) {
        const grid = document.getElementById('vault-grid');
        if (!grid) return;

        grid.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1;">
                <div class="empty-state-icon" style="background: rgba(239, 68, 68, 0.1);">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" style="color: #ef4444;">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                </div>
                <h3 class="empty-state-title">Failed to Load Profiles</h3>
                <p class="empty-state-description">${message}</p>
                <button onclick="vault.refresh()" class="btn btn-primary btn-lg">
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Retry
                </button>
            </div>
        `;
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
}

// Initialize vault
const vault = new ProfileVault();
document.addEventListener('DOMContentLoaded', () => vault.init());
