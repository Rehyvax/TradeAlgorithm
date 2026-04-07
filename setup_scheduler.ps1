# setup_scheduler.ps1
# ====================
# Configura el Task Scheduler de Windows para ejecutar TradeAlgorithm
# automáticamente cada día a las 16:30 hora local.
#
# CÓMO EJECUTAR ESTE SCRIPT:
# ---------------------------
# Opción A — Desde PowerShell con permisos de administrador:
#   1. Busca "PowerShell" en el menú Inicio.
#   2. Haz clic derecho → "Ejecutar como administrador".
#   3. Navega a la carpeta del proyecto:
#        cd "C:\Users\lluis\Downloads\TradeAlgorithm"
#   4. Si la política de ejecución lo bloquea, permítelo para esta sesión:
#        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   5. Ejecuta el script:
#        .\setup_scheduler.ps1
#
# Opción B — Desde CMD con permisos de administrador (una sola línea):
#   powershell -ExecutionPolicy Bypass -File "C:\Users\lluis\Downloads\TradeAlgorithm\setup_scheduler.ps1"
#
# Para verificar que la tarea quedó registrada:
#   Get-ScheduledTask -TaskName "TradeAlgorithm_Daily"
#
# Para eliminar la tarea:
#   Unregister-ScheduledTask -TaskName "TradeAlgorithm_Daily" -Confirm:$false
# ---------------------------

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"

# ── Configuración ──────────────────────────────────────────────────────────
$TaskName   = "TradeAlgorithm_Daily"
$ProjectDir = $PSScriptRoot                          # directorio de este script
$BatFile    = Join-Path $ProjectDir "run_daily.bat"
$LogFile    = Join-Path $ProjectDir "data\live\cron.log"
$RunHour    = 16
$RunMinute  = 30

# ── Validaciones previas ───────────────────────────────────────────────────
if (-not (Test-Path $BatFile)) {
    Write-Error "No se encontró run_daily.bat en: $BatFile"
    exit 1
}

$LogDir = Split-Path $LogFile
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
    Write-Host "Directorio de logs creado: $LogDir"
}

# ── Acción: cmd /c run_daily.bat ───────────────────────────────────────────
# Se usa cmd /c para que el .bat se ejecute en su propio entorno y los
# redireccionamientos >> del .bat funcionen correctamente.
$Action = New-ScheduledTaskAction `
    -Execute  "cmd.exe" `
    -Argument "/c `"$BatFile`"" `
    -WorkingDirectory $ProjectDir

# ── Trigger: cada día a las 16:30 hora local ──────────────────────────────
$Trigger = New-ScheduledTaskTrigger `
    -Daily `
    -At "$($RunHour):$($RunMinute)"

# ── Principal: SYSTEM permite ejecutar sin usuario logueado ───────────────
# BUILTIN\SYSTEM tiene acceso total a archivos locales y no requiere
# que haya ningún usuario activo en la sesión. Si el script Python necesita
# variables de entorno del usuario (p.ej. ALPACA_API_KEY desde .env),
# asegúrate de que el archivo .env esté en el directorio del proyecto;
# daily_execution.py ya usa python-dotenv para cargarlo automáticamente.
$Principal = New-ScheduledTaskPrincipal `
    -UserId    "SYSTEM" `
    -LogonType "ServiceAccount" `
    -RunLevel  "Highest"

# ── Ajustes: reintentos, timeout, condiciones de red ──────────────────────
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit       (New-TimeSpan -Hours 2) `
    -RestartCount             3 `
    -RestartInterval          (New-TimeSpan -Minutes 5) `
    -StartWhenAvailable                                 `
    -RunOnlyIfNetworkAvailable                          `
    -MultipleInstances        IgnoreNew

# Desactivar la condición "ejecutar solo si el equipo está conectado a la red
# eléctrica" para que también funcione en portátiles con batería.
$Settings.DisallowStartIfOnBatteries = $false
$Settings.StopIfGoingOnBatteries     = $false

# ── Registrar (o actualizar) la tarea ─────────────────────────────────────
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    Write-Host "La tarea '$TaskName' ya existe. Actualizando..."
    Set-ScheduledTask `
        -TaskName  $TaskName `
        -Action    $Action `
        -Trigger   $Trigger `
        -Principal $Principal `
        -Settings  $Settings | Out-Null
} else {
    Register-ScheduledTask `
        -TaskName   $TaskName `
        -Action     $Action `
        -Trigger    $Trigger `
        -Principal  $Principal `
        -Settings   $Settings `
        -Description "Ejecuta TradeAlgorithm diariamente a las 16:30 hora local (tras el cierre del mercado americano). Logs en: $LogFile" | Out-Null
}

# ── Confirmación ───────────────────────────────────────────────────────────
$Task = Get-ScheduledTask -TaskName $TaskName
$Info = $Task | Get-ScheduledTaskInfo

Write-Host ""
Write-Host "✓ Tarea configurada correctamente." -ForegroundColor Green
Write-Host ""
Write-Host "  Nombre      : $($Task.TaskName)"
Write-Host "  Estado      : $($Task.State)"
Write-Host "  Próxima ejecución : $($Info.NextRunTime)"
Write-Host "  Comando     : cmd.exe /c `"$BatFile`""
Write-Host "  Directorio  : $ProjectDir"
Write-Host "  Log         : $LogFile"
Write-Host ""
Write-Host "Para ejecutar manualmente ahora mismo:" -ForegroundColor Cyan
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host ""
