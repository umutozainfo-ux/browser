# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_all, collect_data_files

# 1. Collect everything for playwright
p_datas, p_binaries, p_hidden = collect_all('playwright')

# 2. Collect everything for cv2 (OpenCV)
# Use a more explicit collection strategy for OpenCV metadata and config files
cv_datas, cv_binaries, cv_hidden = collect_all('cv2')

# 3. Collect everything for numpy
n_datas, n_binaries, n_hidden = collect_all('numpy')

# 4. Collect greenlet
g_datas, g_binaries, g_hidden = collect_all('greenlet')
# Manually inject _greenlet extension if missing
if not any('_greenlet' in b[0] for b in g_binaries):
    try:
        import greenlet
        import os
        gl_path = os.path.dirname(greenlet.__file__)
        for f in os.listdir(gl_path):
            if f.startswith('_greenlet') and (f.endswith('.pyd') or f.endswith('.so')):
                g_binaries.append((os.path.join(gl_path, f), '.'))
                print(f"DEBUG: Manually added greenlet binary: {f}")
    except Exception as e:
        print(f"DEBUG: Greenlet binary injection failed: {e}")

# 5. Collect aiortc and its heavy dependency 'av' (PyAV)
av_datas, av_binaries, av_hidden = collect_all('av')
rtc_datas, rtc_binaries, rtc_hidden = collect_all('aiortc')

# Combine all
datas = p_datas + cv_datas + n_datas + g_datas + av_datas + rtc_datas
binaries = p_binaries + cv_binaries + n_binaries + g_binaries + av_binaries + rtc_binaries
hiddenimports = p_hidden + cv_hidden + n_hidden + g_hidden + av_hidden + rtc_hidden

# Add specific hidden imports
hiddenimports += [
    'greenlet._greenlet',
    'av._core',
    'av.column',
    'av.container',
    'av.dictionary',
    'av.enum',
    'av.error',
    'av.export',
    'av.filter',
    'av.format',
    'av.frame',
    'av.logging',
    'av.option',
    'av.packet',
    'av.plane',
    'av.stream',
    'av.subtitles',
    'av.utils',
    'av.video',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'engineio.async_drivers.threading',
    # Added for reliability
    'websockets',
    'websockets.legacy',
    'websockets.legacy.client',
    'pystray',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
]

a = Analysis(
    ['node.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Node',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
