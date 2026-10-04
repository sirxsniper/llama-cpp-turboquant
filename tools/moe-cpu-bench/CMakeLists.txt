# [TAG_FN_CPU_MOE_BENCH] CPU expert split benchmark; links only ggml-base and ggml-cpu, so it never loads a GPU backend
set(TARGET moe-cpu-bench)
add_executable(${TARGET} moe-cpu-bench.cpp)
target_link_libraries(${TARGET} PRIVATE ggml-cpu ggml-base ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
