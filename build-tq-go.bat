@echo off
call "%~dp0build-tq-env.bat" >nul
if errorlevel 1 exit /b 1

set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
if "%BUILD_DIR%"=="" set "BUILD_DIR=build-tq-merged"
set "BUILD=%SRC%\%BUILD_DIR%"

if "%1"=="configure" goto :configure
if "%1"=="build" goto :build
if "%1"=="buildall" goto :buildall
if "%1"=="clean" goto :clean
goto :configure

:configure
echo === CONFIGURE ===
cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_COMPRESSION_MODE=size -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -DGGML_NATIVE=ON -DGGML_CPU_REPACK=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=OFF -DGGML_BUILD_TESTS=OFF
if errorlevel 1 (echo [error] cmake configure failed. & exit /b 1)
if "%1"=="configure" exit /b 0

:build
echo === BUILD llama-server ===
cmake --build "%BUILD%" --target llama-server --parallel
if errorlevel 1 (echo [error] build failed. & exit /b 1)
echo === BUILD COMPLETE ===
dir "%BUILD%\bin\llama-server.exe" 2>nul
exit /b 0

:buildall
echo === BUILD ALL TARGETS ===
cmake --build "%BUILD%" --parallel
if errorlevel 1 (echo [error] build failed. & exit /b 1)
echo === BUILD ALL COMPLETE ===
exit /b 0

:clean
if exist "%BUILD%" rmdir /s /q "%BUILD%"
echo Cleaned %BUILD%
exit /b 0
