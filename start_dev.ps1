$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Start-Process -FilePath python `
  -ArgumentList "-m","uvicorn","app.main:app","--host","127.0.0.1","--port","8000" `
  -WorkingDirectory (Join-Path $root "backend") `
  -WindowStyle Hidden

Start-Process -FilePath npm `
  -ArgumentList "run","dev","--","--port","5173" `
  -WorkingDirectory (Join-Path $root "frontend") `
  -WindowStyle Hidden

Start-Sleep -Seconds 4
Write-Host "Backend: http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"
