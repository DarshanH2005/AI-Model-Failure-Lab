param([switch]$SkipTraining)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$externalRoot=Join-Path $env:LOCALAPPDATA 'ShortcutDetective/waterbirds-v1'
$pythonExe=Join-Path $env:LOCALAPPDATA 'ShortcutDetective/venv-gpu/Scripts/python.exe'
Push-Location $projectRoot
try {
    if (-not $SkipTraining) {
        New-Item -ItemType Directory -Force $externalRoot | Out-Null
        if (-not (Test-Path (Join-Path $externalRoot 'waterbird_complete95_forest2water2/metadata.csv'))) {
            if (-not (Test-Path (Join-Path $externalRoot 'waterbirds.tar.gz'))) {
                curl.exe -L --fail --retry 2 --output (Join-Path $externalRoot 'waterbirds.tar.gz') https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz
                if ($LASTEXITCODE -ne 0) { throw 'Official dataset download failed.' }
            }
            & $pythonExe -c "import tarfile; from pathlib import Path; p=Path(r'$externalRoot'); t=tarfile.open(p/'waterbirds.tar.gz'); t.extractall(p,filter='data'); t.close()"
            if ($LASTEXITCODE -ne 0) { throw 'Safe dataset extraction failed.' }
        }
        & $pythonExe scripts/run_waterbirds.py
        if ($LASTEXITCODE -ne 0) { throw 'External study failed; partial results preserved.' }
    }
    $receipt=Get-Content (Join-Path $externalRoot 'release-verification.json') -Raw | ConvertFrom-Json
    if (-not $receipt.passed -or $receipt.runs -ne 6) { throw 'Incomplete external release.' }
    & $pythonExe scripts/Export-Waterbirds.py
    if ($LASTEXITCODE -ne 0) { throw 'External artifact or figure verification failed.' }
    $target=Join-Path $projectRoot 'artifacts/waterbirds-v1'
    New-Item -ItemType Directory -Force $target | Out-Null
    Get-ChildItem -LiteralPath $externalRoot -File | Where-Object { $_.Extension -ne '.gz' } | Copy-Item -Destination $target -Force
    Copy-Item -LiteralPath (Join-Path $externalRoot 'media') -Destination $target -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $externalRoot 'report.md') -Destination (Join-Path $projectRoot 'docs/waterbirds_results.md') -Force
    & $pythonExe -c "from streamlit.testing.v1 import AppTest; a=AppTest.from_file('app/pages/Waterbirds.py').run(timeout=60); assert not a.exception, [e.value for e in a.exception]; assert len(a.dataframe[0].value)==6; print('External dashboard QA passed')"
    if ($LASTEXITCODE -ne 0) { throw 'External dashboard QA failed.' }
    Set-Content -LiteralPath (Join-Path $target 'RELEASE_COMPLETE.txt') -Value 'Six external conditions, artifact verification and dashboard QA complete.' -Encoding utf8
} finally { Pop-Location }


