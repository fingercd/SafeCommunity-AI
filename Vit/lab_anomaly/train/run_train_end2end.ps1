# 在项目根目录执行: .\lab_anomaly\train\run_train_end2end.ps1
Set-Location (Split-Path (Split-Path $PSScriptRoot))
Write-Host "运行: py lab_anomaly/train/train_end2end.py" -ForegroundColor Cyan
py -u lab_anomaly/train/train_end2end.py @args
