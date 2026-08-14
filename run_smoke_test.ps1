param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Model = "dinov2",
    [int]$BatchSize = 4
)

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

& $Python ".\experiments\run_smoke_test.py" --model $Model --batch_size $BatchSize
