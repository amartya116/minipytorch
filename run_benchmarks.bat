@echo off
REM Initialize Intel oneAPI environment
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 >nul

if not exist build\matrix_benchmark.exe (
    echo Error: matrix_benchmark.exe not found!
    echo Run build.bat first to compile benchmarks.
    exit /b 1
)

echo ============================================
echo MiniPyTorch Matrix Operations Benchmarks
echo ============================================
echo.

if "%1"=="" (
    echo Running benchmarks with default output...
    build\matrix_benchmark.exe
) else if "%1"=="--json" (
    echo Running benchmarks and saving to JSON...
    build\matrix_benchmark.exe --benchmark_out=results.json --benchmark_out_format=json
    echo Results saved to results.json
) else if "%1"=="--csv" (
    echo Running benchmarks and saving to CSV...
    build\matrix_benchmark.exe --benchmark_out=results.csv --benchmark_out_format=csv
    echo Results saved to results.csv
) else if "%1"=="--filter" (
    echo Running specific benchmark: %2
    build\matrix_benchmark.exe --benchmark_filter=%2
) else (
    echo Usage:
    echo   run_benchmarks.bat              - Run with default output
    echo   run_benchmarks.bat --json       - Save results as JSON
    echo   run_benchmarks.bat --csv        - Save results as CSV
    echo   run_benchmarks.bat --filter NAME - Run specific benchmark (e.g., BM_MatrixAddition)
    echo.
    echo Example:
    echo   run_benchmarks.bat --filter BM_MatrixAddition
)
