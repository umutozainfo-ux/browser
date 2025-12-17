
"""
Stealth scripts to be injected into Playwright pages.
"""

STEALTH_SCRIPTS = [
    # 1. Mask WebDriver
    """
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined
    });
    """,
    # 2. Mock Chrome Runtime
    """
    window.chrome = {
        runtime: {}
    };
    """,
    # 3. Mask Permissions
    """
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
            Promise.resolve({ state: Notification.permission }) :
            originalQuery(parameters)
    );
    """,
    # 4. WebGL Vendor Spoofing
    """
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(parameter) {
        // UNMASKED_VENDOR_WEBGL
        if (parameter === 37445) {
            return 'Intel Inc.';
        }
        // UNMASKED_RENDERER_WEBGL
        if (parameter === 37446) {
            return 'Intel(R) Iris(R) Xe Graphics';
        }
        return getParameter.apply(this, arguments);
    };
    """
]
