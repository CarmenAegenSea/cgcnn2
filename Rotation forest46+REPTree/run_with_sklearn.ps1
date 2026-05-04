# PowerShell 快捷脚本：在当前工作目录运行 Rotation Forest 基线（sklearn）
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $scriptDir
python run_rotation_forest.py --n_estimators 46 --K 3
Pop-Location
