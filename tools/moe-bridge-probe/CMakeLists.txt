# [TAG_FN_BRIDGE_PROBE] standalone CUDA probe for the MoE doorbell (no model); --selftest [TAG_MOE_BRIDGE] links ggml
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(TARGET moe-bridge-probe)
add_executable(${TARGET} moe-bridge-probe.cu moe-bridge-selftest.cpp) # [TAG_MOE_BRIDGE] --selftest uses ggml
target_link_libraries(${TARGET} PRIVATE ggml CUDA::cudart CUDA::cuda_driver Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
install(TARGETS ${TARGET} RUNTIME)
