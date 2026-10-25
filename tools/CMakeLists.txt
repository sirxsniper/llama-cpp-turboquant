# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(ui)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (LLAMA_CONNECT)
        add_subdirectory(connect)
    endif()
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
        add_subdirectory(moe-cpu-bench) # [TAG_FN_CPU_MOE_BENCH]
    endif()
    add_subdirectory(fit-params)
    if (GGML_METAL)
        add_subdirectory(tuning)
    endif()
    add_subdirectory(results)
    # [TAG_FN_V0252_LINUXBUILD] the probe is Windows-only (Win32 APIs): a Linux build (Docker / NAS) skips it
    if (GGML_CUDA AND WIN32)
        add_subdirectory(moe-bridge-probe) # [TAG_FN_BRIDGE_PROBE]
    endif()
endif()
