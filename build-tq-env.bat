@echo off
REM Set up VS 2022 BuildTools + CUDA 13.1 + CMake + Ninja for llama.cpp TurboQuant build.
REM Source this with: call build-tq-env.bat

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "CUDA_PATH_V13_1=%CUDA_PATH%"

REM VS 2022 BuildTools x64 environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo [error] vcvars64.bat failed.
    exit /b 1
)

REM Prepend CUDA + CMake + Ninja to PATH (after vcvars so its compilers stay first for cl/link)
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%CUDA_PATH%\libnvvp;C:\Program Files\CMake\bin;C:\Users\xSniper\bin;%PATH%"

echo [env] cl    = & where cl
echo [env] nvcc  = & where nvcc
echo [env] cmake = & where cmake
echo [env] ninja = & where ninja
