# [TAG_FN_BRIDGE_PROBE] standalone CUDA probe for the MoE doorbell (no model); --selftest [TAG_MOE_BRIDGE] links ggml
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# ggml-cuda sets CMAKE_CUDA_ARCHITECTURES in its own directory only. Where no default reaches this one (the Windows on Arm
# cross build, cmake/arm64-windows-msvc-cuda.cmake: "CUDA_ARCHITECTURES is empty for target") the probe takes the backend's
if (NOT CMAKE_CUDA_ARCHITECTURES AND TARGET ggml-cuda)
    get_target_property(MOE_BRIDGE_PROBE_CUDA_ARCHS ggml-cuda CUDA_ARCHITECTURES)
    if (MOE_BRIDGE_PROBE_CUDA_ARCHS)
        set(CMAKE_CUDA_ARCHITECTURES "${MOE_BRIDGE_PROBE_CUDA_ARCHS}")
    endif()
endif()

set(TARGET moe-bridge-probe)
add_executable(${TARGET} moe-bridge-probe.cu moe-bridge-selftest.cpp moe-bridge-pin.cpp) # [TAG_MOE_BRIDGE] --selftest uses ggml; [TAG_FN_L22_PIN] test 7
target_link_libraries(${TARGET} PRIVATE ggml CUDA::cudart CUDA::cuda_driver Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
install(TARGETS ${TARGET} RUNTIME)
