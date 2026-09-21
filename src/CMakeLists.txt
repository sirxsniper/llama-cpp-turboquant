llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-kv-tier.cpp        # [TAG_SYNC_CORE_SOURCES] fork-only: turbot tier
    llama-memory.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-moecache.cpp       # [TAG_SYNC_CORE_SOURCES] fork-only
    llama-quant.cpp
    llama-sampler.cpp
    llama-triattention.cpp   # [TAG_SYNC_CORE_SOURCES] fork-only
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

# [TAG_TURBOT_EMBED_PLAN] The built-in turbot plan is compiled in from the checked-in llama-turbot-default-plan.h, which
# docs/turbot/gen_turbot_default_plan.py generates from docs/turbot/plans/turbot-default.plan. The header is checked in,
# not generated here, so build systems that compile src/ without this CMakeLists (vendored copies) still get the plan.
# Drift guard: when the plan file is present, configure stops if the header's SHA-256 of the plan text (LF line
# endings) differs from the file's; both files are configure dependencies, so an edit to either re-runs this check.
set(LLAMA_TURBOT_DEFAULT_PLAN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../docs/turbot/plans/turbot-default.plan")
set(LLAMA_TURBOT_DEFAULT_PLAN_HDR "${CMAKE_CURRENT_SOURCE_DIR}/llama-turbot-default-plan.h")
if (EXISTS "${LLAMA_TURBOT_DEFAULT_PLAN_SRC}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LLAMA_TURBOT_DEFAULT_PLAN_SRC}" "${LLAMA_TURBOT_DEFAULT_PLAN_HDR}")
    file(READ "${LLAMA_TURBOT_DEFAULT_PLAN_SRC}" LLAMA_TURBOT_DEFAULT_PLAN_TEXT)
    string(REPLACE "\r\n" "\n" LLAMA_TURBOT_DEFAULT_PLAN_TEXT "${LLAMA_TURBOT_DEFAULT_PLAN_TEXT}")
    string(SHA256 LLAMA_TURBOT_DEFAULT_PLAN_SHA_SRC "${LLAMA_TURBOT_DEFAULT_PLAN_TEXT}")
    file(STRINGS "${LLAMA_TURBOT_DEFAULT_PLAN_HDR}" LLAMA_TURBOT_DEFAULT_PLAN_SHA_HDR
         REGEX "^#define LLAMA_TURBOT_DEFAULT_PLAN_SHA256 \"[0-9a-f]+\"")
    string(REGEX REPLACE "^#define LLAMA_TURBOT_DEFAULT_PLAN_SHA256 \"([0-9a-f]+)\".*$" "\\1"
           LLAMA_TURBOT_DEFAULT_PLAN_SHA_HDR "${LLAMA_TURBOT_DEFAULT_PLAN_SHA_HDR}")
    if (NOT "${LLAMA_TURBOT_DEFAULT_PLAN_SHA_HDR}" STREQUAL "${LLAMA_TURBOT_DEFAULT_PLAN_SHA_SRC}")
        message(FATAL_ERROR
            "src/llama-turbot-default-plan.h is out of date: it does not match docs/turbot/plans/turbot-default.plan "
            "(sha256 '${LLAMA_TURBOT_DEFAULT_PLAN_SHA_HDR}' in the header, '${LLAMA_TURBOT_DEFAULT_PLAN_SHA_SRC}' for the plan). "
            "Regenerate it with: python docs/turbot/gen_turbot_default_plan.py")
    endif()
endif()

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
