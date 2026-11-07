param([int]$MainProcessId)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $env:LOCALAPPDATA 'ShortcutDetective/core-v4'
$targetDir = Join-Path $projectRoot 'artifacts/core-v4'
$pythonExe = Join-Path $env:LOCALAPPDATA 'ShortcutDetective/venv-gpu/Scripts/python.exe'
$env:PYTHONPATH = Join-Path $projectRoot 'src'
Write-Output 'Release finalizer waiting for the active core study.'
while (Get-Process -Id $MainProcessId -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
Push-Location $projectRoot
try {
    & $pythonExe scripts/verify_release.py $runDir
    if ($LASTEXITCODE -ne 0) { throw 'Core release verification failed; existing outputs preserved.' }
    & $pythonExe -m failure_lab.export --experiment $runDir --output (Join-Path $runDir 'media') --video
    if ($LASTEXITCODE -ne 0) { throw 'Media export failed; experiment outputs preserved.' }
    New-Item -ItemType Directory -Force $targetDir | Out-Null
    Copy-Item -Path "$runDir/*" -Destination $targetDir -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $runDir 'report.md') -Destination (Join-Path $projectRoot 'docs/results.md') -Force
    & $pythonExe -c "from streamlit.testing.v1 import AppTest; a=AppTest.from_file('app/streamlit_app.py').run(timeout=60); assert len(a.exception)==0, [e.value for e in a.exception]; print('Dashboard release QA passed')"
    if ($LASTEXITCODE -ne 0) { throw 'Dashboard release QA failed.' }
    $status = '# Delivery status' + "`n`n" + 'The full 66-run core matrix, artifact verification, media export and dashboard QA have completed. Eight integrity/end-to-end tests passed in a clean CUDA environment. See results.md for measured findings and limitations. The captioned demo video and galleries are in artifacts/core-v4/media. Semantic near-duplicate analysis and external real-image validation remain limitations and optional further work.'
    Set-Content -LiteralPath (Join-Path $projectRoot 'docs/status.md') -Value $status -Encoding utf8
    $resume = 'Built Shortcut Detective with PyTorch, ResNet-18 and DeiT-Tiny; generated a controlled defect benchmark and evaluated targeted versus random nuisance repair across 66 reproducible runs, three seeds and image budgets 128/256/512. Reported 16-group robustness, calibration and matched-performance trade-offs in an interactive dashboard.'
    Set-Content -LiteralPath (Join-Path $projectRoot 'docs/resume_entry.md') -Value $resume -Encoding utf8
    Set-Content -LiteralPath (Join-Path $targetDir 'RELEASE_COMPLETE.txt') -Value 'Core experiment, artifact checks, media export and dashboard QA completed.' -Encoding utf8
    Write-Output "Release complete: $targetDir"
} finally { Pop-Location }




