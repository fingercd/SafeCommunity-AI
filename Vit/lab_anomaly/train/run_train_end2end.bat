@echo off
REM 在项目根目录（含 lab_anomaly 包）下运行
cd /d "%~dp0..\.."
py -u lab_anomaly/train/train_end2end.py %*
