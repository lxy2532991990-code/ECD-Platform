@echo off
cd /d "%~dp0"
echo  ========================================
echo   ECD-Platform Build (PyQt6 Light)
echo  ========================================
echo.
echo [Step 1] Installing dependencies...
python3 -m pip install -r requirements_qt.txt --quiet
python3 -m pip install pyinstaller --quiet
echo [Step 2] Building...
python3 -m PyInstaller --noconfirm --onedir --windowed --name "ECD-Platform" --add-data "ecd_platform;ecd_platform" --hidden-import "matplotlib.backends.backend_qtagg" --hidden-import "matplotlib.backends.backend_agg" --hidden-import "PyQt6.sip" --hidden-import "numpy" --collect-submodules "PyQt6" run_gui.py
echo.
if exist "dist\ECD-Platform" (
    echo  BUILD OK: dist\ECD-Platform\ECD-Platform.exe
) else (
    echo  BUILD FAILED
)
pause
