@echo off
echo Setting up and building the project...
if not exist build mkdir build
cd build
cmake -S .. -B . -Dpybind11_DIR="C:\Users\amart\AppData\Local\Programs\Python\Python314\Lib\site-packages\pybind11\share\cmake\pybind11"
cmake --build . --config Release
cd ..
copy build\Release\example.cp314-win_amd64.pyd example.pyd
echo Build complete! Run with: python -c "import example; print(example.add(1,2))"