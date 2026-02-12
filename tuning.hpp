#pragma once
// tuning.hpp - Centralized tuning parameters for zint library
//
// All algorithm crossover thresholds live here. Determined by benchmarking on Zen 4.
// To retune for a different microarchitecture, run the benchmarks in zint/tests/
// and adjust these constants.

#include <cstdint>

namespace zint {

// ============================================================
// Multiplication thresholds
// ============================================================

// Basecase -> Karatsuba crossover (balanced limb count)
static constexpr uint32_t KARATSUBA_THRESHOLD = 32;

// NTT bi-index dispatch: use NTT when bn >= NTT_BN_MIN AND an*bn >= NTT_AREA.
// NTT_BN_MIN prevents NTT for tiny bn where chunked Karatsuba always wins.
// NTT_AREA captures the crossover curve as a hyperbola an*bn = const.
static constexpr uint32_t NTT_BN_MIN = 128;
static constexpr uint64_t NTT_AREA = 40960;  // ~320*128, ~202*202

// Squaring: basecase exploits symmetry (~1.5x faster), so thresholds differ
static constexpr uint32_t SQR_KARATSUBA_THRESHOLD = 176;
static constexpr uint32_t SQR_NTT_THRESHOLD = 224;

// ============================================================
// Division thresholds
// ============================================================

// Schoolbook -> Newton division crossover (divisor limb count)
static constexpr uint32_t DIV_DC_THRESHOLD = 60;

// Newton reciprocal: base case schoolbook at this many limbs
static constexpr uint32_t INVERT_THRESHOLD = 32;

// ============================================================
// Radix conversion thresholds
// ============================================================

// D&C radix conversion crossover (limb count for to_string, chunk count for from_string)
static constexpr uint32_t RADIX_DC_THRESHOLD = 30;

// Maximum LUT entries for basecase radix conversion (base^lut_t <= this)
static constexpr uint32_t RADIX_LUT_MAX_SIZE = 4096;

} // namespace zint
