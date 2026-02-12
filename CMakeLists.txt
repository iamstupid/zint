cmake_minimum_required(VERSION 3.16)
project(zint LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Options ──────────────────────────────────────────────────────────
option(ZINT_USE_ADX_ASM "Enable ADX/BMI2 ASM kernels (MASM, Windows x64 only)" OFF)

# Auto-enable ASM on MSVC if ml64 is available
if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT DEFINED ZINT_USE_ADX_ASM_SET)
    find_program(ML64_EXECUTABLE ml64)
    if(ML64_EXECUTABLE)
        set(ZINT_USE_ADX_ASM ON CACHE BOOL "" FORCE)
        message(STATUS "zint: ml64 found — enabling ADX ASM kernels")
    endif()
endif()

# ── Header-only interface library ────────────────────────────────────
add_library(zint INTERFACE)
target_include_directories(zint INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(zint INTERFACE cxx_std_17)

if(MSVC)
    target_compile_options(zint INTERFACE /arch:AVX2)
else()
    target_compile_options(zint INTERFACE -mavx2 -mbmi2 -madx)
endif()

# ── ASM kernels (optional, MASM only) ───────────────────────────────
if(ZINT_USE_ADX_ASM)
    if(NOT MSVC)
        message(FATAL_ERROR "ZINT_USE_ADX_ASM requires MSVC + ml64 (MASM)")
    endif()

    enable_language(ASM_MASM)

    set(ZINT_ASM_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/asm/addmul_1_adx.asm
        ${CMAKE_CURRENT_SOURCE_DIR}/asm/submul_1_adx.asm
        ${CMAKE_CURRENT_SOURCE_DIR}/asm/mul_basecase_adx.asm
    )

    add_library(zint_asm STATIC ${ZINT_ASM_SOURCES})
    set_source_files_properties(${ZINT_ASM_SOURCES} PROPERTIES LANGUAGE ASM_MASM)

    target_link_libraries(zint INTERFACE zint_asm)
    target_compile_definitions(zint INTERFACE ZINT_USE_ADX_ASM=1)
    message(STATUS "zint: ADX ASM kernels enabled")
else()
    message(STATUS "zint: ADX ASM kernels disabled (scalar fallback)")
endif()
