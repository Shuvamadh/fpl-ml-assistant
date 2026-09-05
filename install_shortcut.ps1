# install_shortcut.ps1
# Creates/refreshes a Desktop shortcut for the FPL ML Assistant GUI.
# Re-runnable: safe to run again after moving the project or reinstalling Python.

$ErrorActionPreference = 'Stop'

# --- Project root = the folder this script lives in -------------------------
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$EntryPoint  = Join-Path $ProjectRoot 'gui\app.py'
$IconPath    = Join-Path $ProjectRoot 'assets\fpl.ico'

if (-not (Test-Path $EntryPoint)) { throw "Entry point not found: $EntryPoint" }

# --- Resolve the interpreter: prefer pythonw.exe (no console window) --------
$PyDir   = 'C:\Program Files\Python314'
$PythonW = Join-Path $PyDir 'pythonw.exe'
$Python  = Join-Path $PyDir 'python.exe'

if (Test-Path $PythonW) {
    $Target = $PythonW
} elseif (Test-Path $Python) {
    Write-Warning "pythonw.exe not found; falling back to python.exe (a console window will appear)."
    $Target = $Python
} else {
    $cmd = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($null -eq $cmd) { $cmd = Get-Command python.exe -ErrorAction SilentlyContinue }
    if ($null -eq $cmd) { throw "No Python interpreter found at $PyDir or on PATH." }
    $Target = $cmd.Source
}

# --- Icon: use ours if present, else the interpreter's own icon -------------
if (Test-Path $IconPath) {
    $IconLocation = "$IconPath,0"
} else {
    Write-Warning "assets\fpl.ico missing; using the Python executable's own icon."
    $IconLocation = "$Target,0"
}

# --- Resolve the real Desktop (handles OneDrive redirection) ----------------
$Desktop = [Environment]::GetFolderPath('Desktop')
if ([string]::IsNullOrWhiteSpace($Desktop)) { throw "Could not resolve the Desktop folder." }
$LnkPath = Join-Path $Desktop 'FPL Assistant.lnk'

# --- Create the shortcut ----------------------------------------------------
$WScript  = New-Object -ComObject WScript.Shell
$Shortcut = $WScript.CreateShortcut($LnkPath)
$Shortcut.TargetPath       = $Target
$Shortcut.Arguments        = '"' + $EntryPoint + '"'
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.IconLocation     = $IconLocation
$Shortcut.Description      = 'FPL ML Assistant - LightGBM-driven Fantasy Premier League decision support'
$Shortcut.WindowStyle      = 1
$Shortcut.Save()

# --- Verify -----------------------------------------------------------------
if (-not (Test-Path $LnkPath)) { throw "Shortcut was not created at $LnkPath" }
$V = $WScript.CreateShortcut($LnkPath)
Write-Output "Shortcut created and verified:"
Write-Output ("  Path             : {0}" -f $LnkPath)
Write-Output ("  Size             : {0} bytes" -f (Get-Item $LnkPath).Length)
Write-Output ("  TargetPath       : {0}" -f $V.TargetPath)
Write-Output ("  Arguments        : {0}" -f $V.Arguments)
Write-Output ("  WorkingDirectory : {0}" -f $V.WorkingDirectory)
Write-Output ("  IconLocation     : {0}" -f $V.IconLocation)
Write-Output ("  Description      : {0}" -f $V.Description)
