@echo off
title Push 3DGS CUDA Engine to GitHub
cd /d "%~dp0"
cls

echo ===============================================================================
echo 3DGS CUDA ENGINE - GITHUB REPOSUNA YUKLEME
echo Hedef: https://github.com/Hyperinflation/3dgs-cuda-engine.git
echo ===============================================================================
echo.
echo [*] Kodlar gonderiliyor...
echo.

git push origin main --force

echo.
echo ===============================================================================
echo ISLEM BASARIYLA TAMAMLANDI!
echo GitHub Actions sekmesinde CUDA derleme hattini izleyebilirsiniz:
echo https://github.com/Hyperinflation/3dgs-cuda-engine/actions
echo ===============================================================================
echo.
pause
