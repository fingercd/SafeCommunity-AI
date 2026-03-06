# 已知异常 MIL 分类器训练 - 启动脚本
# 用法：在 PowerShell 中执行 .\run_train_known_classifier.ps1
# 或在资源管理器中右键此文件 -> 使用 PowerShell 运行

$VitRoot = "c:\Users\Administrator\Desktop\Vit"
Set-Location $VitRoot
$env:PYTHONPATH = $VitRoot
$env:PYTHONUNBUFFERED = "1"

Write-Host "工作目录: $VitRoot" -ForegroundColor Cyan
Write-Host "运行: py lab_anomaly/train/train_known_classifier.py" -ForegroundColor Cyan
Write-Host ""

py -u lab_anomaly/train/train_known_classifier.py @args

if ($LASTEXITCODE -ne 0) {
    Write-Host "退出码: $LASTEXITCODE" -ForegroundColor Red
    pause
}
