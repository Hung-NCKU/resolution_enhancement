@echo off
chcp 65001 >nul
title AI 影片畫質修復
rem 以本檔所在資料夾為專案目錄（不寫死路徑）
echo 啟動中，完成後會自動開啟瀏覽器 http://localhost:7860
start "" /b powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){try{Invoke-WebRequest http://localhost:7860 -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process http://localhost:7860; break}catch{Start-Sleep 1}}"
wsl -d Ubuntu-24.04 --cd "%~dp0." -- bash ./run.sh
pause
