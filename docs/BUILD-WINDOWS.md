# Windows build guide

This is the verified-working recipe for building this fork on Windows 11
x64 with CUDA. Tested 2026-05-05 with VS 2022 BuildTools 14.44.35207,
CUDA 13.1.80, CMake 4.3.2, Ninja 1.13.2 on a 5090 plus 5060 Ti rig.

If you only need binaries, grab them from the
[HuggingFace release](https://huggingface.co/sirxsniper/llama-cpp-turboquant-binaries).
No build required.

---

## 1. Prerequisites

| Tool | Version | Source |
|---|---|---|
| Visual Studio 2022 BuildTools | 17.x with Workload "Desktop development with C++", MSVC v143, Windows 11 SDK | `winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --add Microsoft.VisualStudio.Component.VC.CMake.Project --includeRecommended"` |
| CUDA Toolkit 13.1 | 13.1.x. NOT 13.2. The fork has only been tested against 13.1, 13.2 is untested. | [archive download](https://developer.nvidia.com/cuda-13-1-0-download-archive). Pick *Windows / x86_64 / 11 / exe (local)*. winget's `Nvidia.CUDA` resolves to 13.2 and will not give you 13.1. |
| CMake | 3.27 or newer | `winget install Kitware.CMake` |
| Ninja | 1.11 or newer | `winget install Ninja-build.Ninja` (needs admin), or grab `ninja-win.zip` from [releases](https://github.com/ninja-build/ninja/releases/latest) and drop `ninja.exe` somewhere on `PATH` |
| NVIDIA driver | 596.x or newer (anything that exposes CUDA 13.1) | already installed if you can run `nvidia-smi` |
| git | any | `winget install Git.Git` |

> Why pin CUDA 13.1? The sm_120 (Blackwell) toolchain matured in 13.1 and
> the `__dp4a` and FA-MMA paths used by turbo were last validated against
> it. 13.2 changes parts of the cooperative-groups API and we have not
> retested. If you have to use a different version, build and run the
> test suite (`ctest` in the build dir) before relying on the result.

### Verify your install

After all four tools are in place, in a fresh shell:

```cmd
cl                          REM should print "Microsoft (R) C/C++ Optimizing Compiler"
nvcc --version              REM "Cuda compilation tools, release 13.1"
cmake --version             REM 3.27 or newer
ninja --version             REM 1.11 or newer
```

If `cl` is not found, you need to source the VS x64 environment. See
section 3 below.

### Common gotchas

* CUDA 13.1 silent install with a hand-picked component list (`-s nvcc_13.1
  cudart_13.1 ...`) drops `crt/host_config.h` and the compile fails with
  `Cannot open include file: 'crt/host_config.h'`. Either install with the
  default selection (`cuda_13.1.0_windows.exe -s` with no component list)
  or add `cudart_dev_13.1` and the rest of the dev components yourself.

* CUDA 13.x relocates the runtime DLLs to `bin\x64\` (used to be `bin\`).
  This is normal. `lib\x64\` holds the import libs as before. Add both
  `bin` and `bin\x64` to `PATH` for builds that link against the
  redistribs.

* VS BuildTools install via `winget` followed immediately by use in the
  same shell does not work. The new compiler is on `PATH` only in newly
  spawned shells. The existing shell has the pre-install environment.
  Open a fresh terminal after the install completes.

* Ninja `winget install --scope machine` requires admin elevation. If you
  see exit code 25, either use `--scope user` or fall back to the
  GitHub-releases zip (just drop the .exe somewhere on `PATH`).

---

## 2. Clone

```cmd
git clone https://github.com/sirxsniper/llama-cpp-turboquant.git
cd llama-cpp-turboquant
```

If you cloned this elsewhere first (for example via WSL) and Windows
refuses to operate on it ("dubious ownership" error), add a safe-directory
exception:

```cmd
git config --global --add safe.directory "C:/path/to/llama-cpp-turboquant"
```

---

## 3. Configure and build

Open the **x64 Native Tools Command Prompt for VS 2022** (Start menu).
That gives you a shell with `cl.exe` already on `PATH`. From there:

```cmd
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%PATH%"

cmake -S . -B build-tq -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=120a ^
  -DGGML_CUDA=ON ^
  -DGGML_CUDA_FA=ON ^
  -DGGML_CUDA_FA_ALL_QUANTS=ON ^
  -DGGML_CUDA_GRAPHS=ON ^
  -DGGML_CUDA_COMPRESSION_MODE=size ^
  -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 ^
  -DGGML_NATIVE=ON ^
  -DGGML_CPU_REPACK=ON ^
  -DLLAMA_BUILD_EXAMPLES=ON ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DGGML_BUILD_TESTS=OFF

cmake --build build-tq --parallel
```

If you don't have the x64 Native Tools shortcut, source the VS env from
any plain cmd shell with:

```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

before the `cmake` lines above.

### Picking `CMAKE_CUDA_ARCHITECTURES`

| GPU | Use |
|---|---|
| RTX 5090, 5080, 5070, 5070 Ti, 5060 Ti | `120a` (Blackwell consumer) |
| RTX 4090, 4080, 4070 (Ada) | `89` |
| RTX 3090, 3080, 3070 (Ampere) | `86` |
| RTX 2080, 2070, 2060 (Turing) | `75` |
| Multiple of the above | semi-colon list, for example `89;120a` |

The fork's CMake auto-detects via `GGML_NATIVE=ON`, but pinning explicitly
gives a smaller binary and a faster compile.

---

## 4. Output and runtime

The build produces all artefacts under `build-tq\bin\`. The minimum set
you need to run a server is:

```
llama-server.exe
llama.dll
llama-common.dll
mtmd.dll
ggml.dll
ggml-base.dll
ggml-cpu.dll
ggml-cuda.dll
```

Plus the CUDA runtime redistribs (loaded from your CUDA install via
`PATH`):

```
cudart64_13.dll       (in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\)
cublas64_13.dll
cublasLt64_13.dll
```

Quick smoke test:

```cmd
build-tq\bin\llama-server.exe --version
```

You should see something like:

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 32606 MiB):
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes, VRAM: 32606 MiB
version: 759 (<commit>)
built with MSVC 19.44.35226.0 for Windows AMD64
```

---

## 5. Things that look broken but aren't

When you first run a build under PowerShell, you may see noisy errors
like:

```
'vswhere.exe' is not recognized as an internal or external command
```

That's `vcvars64.bat` calling `vswhere` to discover the install. It
succeeds even though stderr leaks the message. As long as `cl` resolves
afterwards, the env is fine.

```
NativeCommandError ... (ggml_cuda_init: found 2 CUDA devices ...)
```

PowerShell wraps any stderr from a native command as an "error" object.
The process exit code is 0. It's just diagnostic output from llama.cpp.

```
NCCL not found, performance for multiple CUDA GPUs will be suboptimal
```

NCCL has no Windows port and has never been used by this fork. The
warning is harmless. Multi-GPU still works via the standard CUDA
peer-to-peer path.

```
OpenSSL not found, HTTPS support disabled
```

The webui still works fine on plain HTTP. If you specifically need HTTPS,
install the OpenSSL Windows binaries from
<https://slproweb.com/products/Win32OpenSSL.html> and re-configure.

---

## 6. Performance tuning notes

* Windows power plan must be High Performance or Ultimate Performance.
  The default Balanced plan keeps the GPU pstate one notch below P0
  during generation, costing roughly 10% TG t/s. Switch via `powercfg.cpl`
  or `powercfg /setactive SCHEME_MIN`.

* NVIDIA Control Panel, 3D Settings, Power management mode: "Prefer
  maximum performance", either globally or for `llama-server.exe`
  specifically. Same effect as the power plan.

* Driver and CUDA-context state can drift. If TG perf mysteriously drops
  about 10% after a long working session (lots of model loads and
  unloads), a reboot is the cheapest first move. This was needed once
  during this round of work.

* `--no-mmap` (or `-mmp 0` for `llama-bench`) is recommended once the
  model fits in VRAM. Saves a small TG hit from page-fault stalls when
  the OS lazily pages in mmapped weights.

* PCIe x8 vs x16 turns out not to matter for token generation (model is
  GPU-resident, embedding lookups are tiny). Don't burn time on this if
  you see "PCIe Gen 5 x8" under load. That's normal for two-GPU systems
  on most consumer mobos.
