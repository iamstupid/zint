cmake_minimum_required(VERSION 3.16)
project(zint LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Header-only interface library ────────────────────────────────────
add_library(zint INTERFACE)
target_include_directories(zint INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(zint INTERFACE cxx_std_17)

if(MSVC)
    target_compile_options(zint INTERFACE /arch:AVX2)
else()
    target_compile_options(zint INTERFACE -mavx2 -mbmi2 -madx)
endif()

# ── ASM kernels (required) ──────────────────────────────────────────
# MASM (.asm) on Windows/MSVC, GAS (.S) on Linux/macOS with GCC or Clang.

if(MSVC)
    enable_language(ASM_MASM)

    set(ZINT_ASM_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/asm/addmul_1_adx.asm
        ${CMAKE_CURRENT_SOURCE_DIR}/asm/submul_1_adx.asm
        ${CMAKE_CURRENT_SOURCE_DIR}/asm/mul_basecase_adx.asm
    )
    add_library(zint_asm STATIC ${ZINT_ASM_SOURCES})
    set_source_files_properties(${ZINT_ASM_SOURCES} PROPERTIES LANGUAGE ASM_MASM)

    message(STATUS "zint: ASM kernels via MASM (Windows x64)")
else()
    enable_language(ASM)

    set(ZINT_ASM_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/asm/addmul_1_adx.S
        ${CMAKE_CURRENT_SOURCE_DIR}/asm/submul_1_adx.S
        ${CMAKE_CURRENT_SOURCE_DIR}/asm/mul_basecase_adx.S
    )
    add_library(zint_asm STATIC ${ZINT_ASM_SOURCES})
    set_source_files_properties(${ZINT_ASM_SOURCES} PROPERTIES
        LANGUAGE ASM
        COMPILE_FLAGS "-mavx2 -mbmi2 -madx"
    )

    message(STATUS "zint: ASM kernels via GAS (System V ABI)")
endif()

target_link_libraries(zint INTERFACE zint_asm)
target_compile_definitions(zint INTERFACE ZINT_USE_ADX_ASM=1)
