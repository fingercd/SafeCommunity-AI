# ECVA 冒烟流水线：请在 vlm 项目根目录执行，或先 cd 到此脚本的上级目录
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Py = "C:\ProgramData\anaconda3\envs\yolovv\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
    if (Test-Path ".venv\Scripts\python.exe") { $Py = ".\.venv\Scripts\python.exe" }
}

& $Py data/parse_ecva.py --config configs/quick_2b.yaml
& $Py data/prepare_clips.py --mode ecva --config configs/quick_2b.yaml --clip_len 16
& $Py data/build_dataset.py --source ecva --config configs/quick_2b.yaml
& $Py train/train_qlora.py --config configs/quick_2b.yaml

Write-Host "完成。可选评测: $Py train/evaluate.py --config configs/quick_2b.yaml --model_path outputs/qlora_2b/final"
