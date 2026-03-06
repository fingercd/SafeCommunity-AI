@echo off
chcp 65001 >nul
set VIT_ROOT=c:\Users\Administrator\Desktop\Vit
set PYTHONPATH=%VIT_ROOT%
set PYTHONUNBUFFERED=1
cd /d "%VIT_ROOT%"
echo 工作目录: %VIT_ROOT%
echo 运行训练脚本...
echo.
py -u lab_anomaly/train/train_known_classifier.py %*
if errorlevel 1 pause
