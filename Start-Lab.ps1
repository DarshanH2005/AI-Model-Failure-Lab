param(
    [ValidateSet('dashboard','smoke','pilot','core','test','generate')]
    [string]$Mode = 'dashboard'
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$runRoot = Join-Path $env:LOCALAPPDATA 'ShortcutDetective'
New-Item -ItemType Directory -Force $runRoot | Out-Null
$pythonExe = 'python'
$gpuPython = Join-Path $env:LOCALAPPDATA 'ShortcutDetective/venv-gpu/Scripts/python.exe'
if (Test-Path -LiteralPath $gpuPython) { $pythonExe = $gpuPython }
$env:PYTHONPATH = Join-Path $projectRoot 'src'
Push-Location $projectRoot
try {
    if ($Mode -eq 'dashboard') {
        & $pythonExe -m streamlit run app/streamlit_app.py --server.headless true --browser.gatherUsageStats false
    } elseif ($Mode -eq 'test') {
        & $pythonExe -m pytest -q -p no:cacheprovider
    } elseif ($Mode -eq 'generate') {
        & $pythonExe -c "from failure_lab.cli import load_config; from failure_lab.data import generate; import sys; generate(load_config('configs/smoke.yaml'),sys.argv[1])" (Join-Path $runRoot 'generated-smoke')
    } else {
        $config = switch ($Mode) { 'smoke' {'configs/smoke.yaml'} 'pilot' {'configs/pilot.yaml'} 'core' {'configs/experiments.yaml'} }
        $experiment = switch ($Mode) { 'smoke' {'smoke-v1'} 'pilot' {'pilot-resnet-v1'} 'core' {'core-v4'} }
        $env:FAILURE_LAB_RENDER_CACHE = if ($Mode -eq 'core') { '512' } else { '256' }
        $output = Join-Path $runRoot $experiment
        & $pythonExe -m failure_lab.cli run-experiments --config $config --root $output
        if ($LASTEXITCODE -ne 0) { throw "Experiment failed ($LASTEXITCODE); partial outputs preserved in $output" }
        $destination = Join-Path $projectRoot "artifacts/$experiment"
        New-Item -ItemType Directory -Force $destination | Out-Null
        Copy-Item -Path "$output/*" -Destination $destination -Force
    }
} finally { Pop-Location }





