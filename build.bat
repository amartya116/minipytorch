@echo off
echo Building with Intel oneAPI Compiler...

REM Initialize Intel oneAPI environment
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64

REM Clean old build to prevent cache mismatch
if exist build rmdir /s /q build
if exist CMakeCache.txt del CMakeCache.txt
mkdir build
cd build

REM Configure with Intel oneAPI compiler (icx)
cmake -S .. -B . -G "Ninja" -DCMAKE_CXX_COMPILER=icx -Dpybind11_DIR="C:\Users\amart\AppData\Local\Programs\Python\Python314\Lib\site-packages\pybind11\share\cmake\pybind11"

if errorlevel 1 (
    echo CMake configuration failed!
    cd ..
    exit /b 1
)

REM Build the project
cmake --build . --config Release || cmake --build .

if errorlevel 1 (
    echo Build failed!
    cd ..
    exit /b 1
)

cd ..
if exist build\Release\example.cp314-win_amd64.pyd (
    copy build\Release\example.cp314-win_amd64.pyd example.pyd
)
echo Build complete!
echo.
echo To run benchmarks:
echo   build\matrix_benchmark.exe
echo.
echo To run with specific options:
echo   build\matrix_benchmark.exe --benchmark_out=results.json --benchmark_out_format=json 