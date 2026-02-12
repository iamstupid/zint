#pragma once
// mpn.hpp - Low-level limb array operations for arbitrary-precision integers
//
// GMP-compatible naming: mpn_add_n, mpn_sub_n, mpn_mul_1, etc.
// All operations are on unsigned limb arrays in little-endian order.
// Uses compiler intrinsics for carry-chain arithmetic (adc/sbb/mulq).

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <immintrin.h>

// ============================================================
// Force-inline / noinline macros
// ============================================================

#if defined(_MSC_VER)
  #define ZINT_FORCEINLINE __forceinline
  #define ZINT_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
  #define ZINT_FORCEINLINE [[gnu::always_inline]] inline
  #define ZINT_NOINLINE __attribute__((noinline))
#else
  #define ZINT_FORCEINLINE inline
  #define ZINT_NOINLINE
#endif

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_addcarry_u64, _subborrow_u64, _umul128, _BitScanReverse64, _udiv128)
#else
#include <x86intrin.h>
#endif

namespace zint {

using limb_t = uint64_t;
using slimb_t = int64_t;
static constexpr int LIMB_BITS = 64;

// ============================================================
// Portable 128-bit arithmetic wrappers
// ============================================================

// 64x64 -> 128-bit multiply: returns low 64 bits, *hi = high 64 bits
inline limb_t umul_hilo(limb_t a, limb_t b, limb_t* hi) {
#ifdef _MSC_VER
    return _umul128(a, b, hi);
#else
    unsigned __int128 r = (unsigned __int128)a * b;
    *hi = (limb_t)(r >> 64);
    return (limb_t)r;
#endif
}

// 128 / 64 -> 64-bit quotient + 64-bit remainder
// Precondition: hi < d (quotient fits in 64 bits)
inline limb_t udiv_128(limb_t hi, limb_t lo, limb_t d, limb_t* rem) {
#ifdef _MSC_VER
    return _udiv128(hi, lo, d, rem);
#else
    unsigned __int128 n = ((unsigned __int128)hi << 64) | lo;
    *rem = (limb_t)(n % d);
    return (limb_t)(n / d);
#endif
}

// Count leading zeros (undefined for x == 0)
inline int clz64(limb_t x) {
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanReverse64(&idx, x);
    return 63 - (int)idx;
#else
    return __builtin_clzll(x);
#endif
}

// ============================================================
// Memory operations
// ============================================================

// Zero n limbs
inline void mpn_zero(limb_t* rp, uint32_t n) {
    std::memset(rp, 0, (size_t)n * sizeof(limb_t));
}

// Copy n limbs, increasing address (safe when rp <= ap)
inline void mpn_copyi(limb_t* rp, const limb_t* ap, uint32_t n) {
    std::memmove(rp, ap, (size_t)n * sizeof(limb_t));
}

// Copy n limbs, decreasing address (safe when rp >= ap)
inline void mpn_copyd(limb_t* rp, const limb_t* ap, uint32_t n) {
    std::memmove(rp, ap, (size_t)n * sizeof(limb_t));
}

// ============================================================
// Bitwise ops (on limb arrays)
// ============================================================

inline void mpn_not_n(limb_t* rp, const limb_t* ap, uint32_t n) {
    uint32_t i = 0;
    const __m256i ones = _mm256_set1_epi64x(-1);
    for (; i + 4 <= n; i += 4) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(ap + i));
        __m256i r = _mm256_xor_si256(a, ones);
        _mm256_storeu_si256((__m256i*)(rp + i), r);
    }
    for (; i < n; ++i) rp[i] = ~ap[i];
}

inline void mpn_and_n(limb_t* rp, const limb_t* ap, const limb_t* bp, uint32_t n) {
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(ap + i));
        __m256i b = _mm256_loadu_si256((const __m256i*)(bp + i));
        __m256i r = _mm256_and_si256(a, b);
        _mm256_storeu_si256((__m256i*)(rp + i), r);
    }
    for (; i < n; ++i) rp[i] = ap[i] & bp[i];
}

inline void mpn_or_n(limb_t* rp, const limb_t* ap, const limb_t* bp, uint32_t n) {
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(ap + i));
        __m256i b = _mm256_loadu_si256((const __m256i*)(bp + i));
        __m256i r = _mm256_or_si256(a, b);
        _mm256_storeu_si256((__m256i*)(rp + i), r);
    }
    for (; i < n; ++i) rp[i] = ap[i] | bp[i];
}

inline void mpn_xor_n(limb_t* rp, const limb_t* ap, const limb_t* bp, uint32_t n) {
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(ap + i));
        __m256i b = _mm256_loadu_si256((const __m256i*)(bp + i));
        __m256i r = _mm256_xor_si256(a, b);
        _mm256_storeu_si256((__m256i*)(rp + i), r);
    }
    for (; i < n; ++i) rp[i] = ap[i] ^ bp[i];
}

// ============================================================
// Addition
// ============================================================

// rp[0..n) = ap[0..n) + bp[0..n), return carry (0 or 1)
inline limb_t mpn_add_n(limb_t* rp, const limb_t* ap, const limb_t* bp, uint32_t n) {
    unsigned char carry = 0;
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        carry = _addcarry_u64(carry, ap[i],   bp[i],   (unsigned long long*)&rp[i]);
        carry = _addcarry_u64(carry, ap[i+1], bp[i+1], (unsigned long long*)&rp[i+1]);
        carry = _addcarry_u64(carry, ap[i+2], bp[i+2], (unsigned long long*)&rp[i+2]);
        carry = _addcarry_u64(carry, ap[i+3], bp[i+3], (unsigned long long*)&rp[i+3]);
    }
    for (; i < n; i++) {
        carry = _addcarry_u64(carry, ap[i], bp[i], (unsigned long long*)&rp[i]);
    }
    return carry;
}

// rp[0..n) = ap[0..n) + b, return carry
// Carry only propagates through 0xFFFF...FFFF limbs; SIMD scans 4 at a time.
inline limb_t mpn_add_1(limb_t* rp, const limb_t* ap, uint32_t n, limb_t b) {
    assert(n > 0);
    unsigned char carry = _addcarry_u64(0, ap[0], b, (unsigned long long*)&rp[0]);
    uint32_t i = 1;
    if (carry) {
        const __m256i v_max = _mm256_set1_epi64x(-1LL);
        for (; i + 4 <= n; i += 4) {
            __m256i v = _mm256_loadu_si256((const __m256i*)(ap + i));
            if (_mm256_movemask_epi8(_mm256_cmpeq_epi64(v, v_max)) != -1) break;
            // All 4 limbs are MAX → +1 wraps to 0, carry continues
            _mm256_storeu_si256((__m256i*)(rp + i), _mm256_setzero_si256());
        }
        for (; i < n; ++i) {
            carry = _addcarry_u64(carry, ap[i], 0, (unsigned long long*)&rp[i]);
            if (!carry) { ++i; break; }
        }
    }
    if (rp != ap) {
        for (; i < n; ++i) rp[i] = ap[i];
    }
    return (limb_t)carry;
}

// Version that always writes all n output limbs.
// Same SIMD carry-propagation scan as mpn_add_1.
inline limb_t mpn_add_1v(limb_t* rp, const limb_t* ap, uint32_t n, limb_t b) {
    assert(n > 0);
    unsigned char carry = _addcarry_u64(0, ap[0], b, (unsigned long long*)&rp[0]);
    uint32_t i = 1;
    if (carry) {
        const __m256i v_max = _mm256_set1_epi64x(-1LL);
        for (; i + 4 <= n; i += 4) {
            __m256i v = _mm256_loadu_si256((const __m256i*)(ap + i));
            if (_mm256_movemask_epi8(_mm256_cmpeq_epi64(v, v_max)) != -1) break;
            _mm256_storeu_si256((__m256i*)(rp + i), _mm256_setzero_si256());
        }
        for (; i < n; ++i) {
            carry = _addcarry_u64(carry, ap[i], 0, (unsigned long long*)&rp[i]);
            if (!carry) { ++i; break; }
        }
    }
    // Copy remaining
    if (rp != ap) {
        for (; i < n; ++i) rp[i] = ap[i];
    }
    return carry;
}

// rp[0..an) = ap[0..an) + bp[0..bn), return carry
// Precondition: an >= bn > 0
inline limb_t mpn_add(limb_t* rp, const limb_t* ap, uint32_t an, const limb_t* bp, uint32_t bn) {
    assert(an >= bn && bn > 0);
    limb_t carry = mpn_add_n(rp, ap, bp, bn);
    if (an > bn) {
        if (carry) {
            carry = mpn_add_1v(rp + bn, ap + bn, an - bn, carry);
        } else if (rp != ap) {
            mpn_copyi(rp + bn, ap + bn, an - bn);
        }
    }
    return carry;
}

// rp[0..n) = ap[0..n) + (bp[0..n) << cnt), return carry limb (0..2^cnt)
// Precondition: n > 0, 0 < cnt < 64. rp may alias ap; rp must not alias bp.
inline limb_t mpn_addlsh_n(limb_t* rp, const limb_t* ap, const limb_t* bp, uint32_t n, unsigned cnt) {
    assert(n > 0 && cnt > 0 && cnt < LIMB_BITS);
    unsigned rcnt = LIMB_BITS - cnt;
    limb_t sh_carry = 0;
    unsigned char carry = 0;
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        limb_t b0 = bp[i], b1 = bp[i+1], b2 = bp[i+2], b3 = bp[i+3];
        limb_t s0 = (b0 << cnt) | sh_carry;
        limb_t s1 = (b1 << cnt) | (b0 >> rcnt);
        limb_t s2 = (b2 << cnt) | (b1 >> rcnt);
        limb_t s3 = (b3 << cnt) | (b2 >> rcnt);
        sh_carry = b3 >> rcnt;
        carry = _addcarry_u64(carry, ap[i],   s0, (unsigned long long*)&rp[i]);
        carry = _addcarry_u64(carry, ap[i+1], s1, (unsigned long long*)&rp[i+1]);
        carry = _addcarry_u64(carry, ap[i+2], s2, (unsigned long long*)&rp[i+2]);
        carry = _addcarry_u64(carry, ap[i+3], s3, (unsigned long long*)&rp[i+3]);
    }
    for (; i < n; ++i) {
        limb_t b = bp[i];
        limb_t shifted = (b << cnt) | sh_carry;
        sh_carry = b >> rcnt;
        carry = _addcarry_u64(carry, ap[i], shifted, (unsigned long long*)&rp[i]);
    }
    return sh_carry + (limb_t)carry;
}

// ============================================================
// Subtraction
// ============================================================

// rp[0..n) = ap[0..n) - bp[0..n), return borrow (0 or 1)
inline limb_t mpn_sub_n(limb_t* rp, const limb_t* ap, const limb_t* bp, uint32_t n) {
    unsigned char borrow = 0;
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        borrow = _subborrow_u64(borrow, ap[i],   bp[i],   (unsigned long long*)&rp[i]);
        borrow = _subborrow_u64(borrow, ap[i+1], bp[i+1], (unsigned long long*)&rp[i+1]);
        borrow = _subborrow_u64(borrow, ap[i+2], bp[i+2], (unsigned long long*)&rp[i+2]);
        borrow = _subborrow_u64(borrow, ap[i+3], bp[i+3], (unsigned long long*)&rp[i+3]);
    }
    for (; i < n; i++) {
        borrow = _subborrow_u64(borrow, ap[i], bp[i], (unsigned long long*)&rp[i]);
    }
    return borrow;
}

// rp[0..n) = ap[0..n) - b, return borrow
// Borrow only propagates through 0x0000...0000 limbs; SIMD scans 4 at a time.
inline limb_t mpn_sub_1(limb_t* rp, const limb_t* ap, uint32_t n, limb_t b) {
    assert(n > 0);
    unsigned char borrow = _subborrow_u64(0, ap[0], b, (unsigned long long*)&rp[0]);
    uint32_t i = 1;
    if (borrow) {
        const __m256i v_zero = _mm256_setzero_si256();
        const __m256i v_max  = _mm256_set1_epi64x(-1LL);
        for (; i + 4 <= n; i += 4) {
            __m256i v = _mm256_loadu_si256((const __m256i*)(ap + i));
            if (_mm256_movemask_epi8(_mm256_cmpeq_epi64(v, v_zero)) != -1) break;
            // All 4 limbs are 0 → 0-1 wraps to MAX, borrow continues
            _mm256_storeu_si256((__m256i*)(rp + i), v_max);
        }
        for (; i < n; ++i) {
            borrow = _subborrow_u64(borrow, ap[i], 0, (unsigned long long*)&rp[i]);
            if (!borrow) { ++i; break; }
        }
    }
    if (rp != ap) {
        for (; i < n; ++i) rp[i] = ap[i];
    }
    return (limb_t)borrow;
}

// rp[0..an) = ap[0..an) - bp[0..bn), return borrow
// Precondition: an >= bn > 0
inline limb_t mpn_sub(limb_t* rp, const limb_t* ap, uint32_t an, const limb_t* bp, uint32_t bn) {
    assert(an >= bn && bn > 0);
    limb_t borrow = mpn_sub_n(rp, ap, bp, bn);
    if (an > bn) {
        if (borrow) {
            borrow = mpn_sub_1(rp + bn, ap + bn, an - bn, borrow);
        } else if (rp != ap) {
            mpn_copyi(rp + bn, ap + bn, an - bn);
        }
    }
    return borrow;
}

// ============================================================
// Shift
// ============================================================

// rp[0..n) = ap[0..n) << cnt, return bits shifted out from top
// Precondition: 0 < cnt < 64; n > 0
// Direction: processes high to low (safe for rp == ap or rp > ap)
// AVX2: processes 4 limbs per iteration using overlapping loads.
inline limb_t mpn_lshift(limb_t* rp, const limb_t* ap, uint32_t n, unsigned cnt) {
    assert(cnt > 0 && cnt < LIMB_BITS && n > 0);
    unsigned rcnt = LIMB_BITS - cnt;
    limb_t retval = ap[n - 1] >> rcnt;

    __m128i vcnt  = _mm_cvtsi32_si128(cnt);
    __m128i vrcnt = _mm_cvtsi32_si128(rcnt);

    // Process 4 limbs at a time, high to low.
    // rp[i-3..i] = (ap[i-3..i] << cnt) | (ap[i-4..i-1] >> rcnt)
    uint32_t i = n - 1;
    for (; i >= 4; i -= 4) {
        __m256i a_cur  = _mm256_loadu_si256((const __m256i*)(ap + i - 3));
        __m256i a_prev = _mm256_loadu_si256((const __m256i*)(ap + i - 4));
        __m256i hi = _mm256_sll_epi64(a_cur, vcnt);
        __m256i lo = _mm256_srl_epi64(a_prev, vrcnt);
        _mm256_storeu_si256((__m256i*)(rp + i - 3), _mm256_or_si256(hi, lo));
    }
    // Scalar cleanup
    for (; i > 0; i--) {
        rp[i] = (ap[i] << cnt) | (ap[i - 1] >> rcnt);
    }
    rp[0] = ap[0] << cnt;
    return retval;
}

// rp[0..n) = ap[0..n) >> cnt, return bits shifted out from bottom
// Precondition: 0 < cnt < 64; n > 0
// Direction: processes low to high (safe for rp == ap or rp < ap)
// AVX2: processes 4 limbs per iteration using overlapping loads.
inline limb_t mpn_rshift(limb_t* rp, const limb_t* ap, uint32_t n, unsigned cnt) {
    assert(cnt > 0 && cnt < LIMB_BITS && n > 0);
    unsigned rcnt = LIMB_BITS - cnt;
    limb_t retval = ap[0] << rcnt;

    __m128i vcnt  = _mm_cvtsi32_si128(cnt);
    __m128i vrcnt = _mm_cvtsi32_si128(rcnt);

    // Process 4 limbs at a time, low to high.
    // rp[i..i+3] = (ap[i..i+3] >> cnt) | (ap[i+1..i+4] << rcnt)
    uint32_t i = 0;
    for (; i + 4 < n; i += 4) {
        __m256i a_cur  = _mm256_loadu_si256((const __m256i*)(ap + i));
        __m256i a_next = _mm256_loadu_si256((const __m256i*)(ap + i + 1));
        __m256i lo = _mm256_srl_epi64(a_cur, vcnt);
        __m256i hi = _mm256_sll_epi64(a_next, vrcnt);
        _mm256_storeu_si256((__m256i*)(rp + i), _mm256_or_si256(lo, hi));
    }
    // Scalar cleanup
    for (; i + 1 < n; i++) {
        rp[i] = (ap[i] >> cnt) | (ap[i + 1] << rcnt);
    }
    rp[n - 1] = ap[n - 1] >> cnt;
    return retval;
}

// ============================================================
// Comparison
// ============================================================

// Compare ap[0..n) with bp[0..n). Returns -1, 0, or 1.
// AVX2: compares 4 limbs at a time from the top; BSR finds highest differing lane.
inline int mpn_cmp(const limb_t* ap, const limb_t* bp, uint32_t n) {
    uint32_t i = n;
    for (; i >= 4; i -= 4) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(ap + i - 4));
        __m256i b = _mm256_loadu_si256((const __m256i*)(bp + i - 4));
        int mask = _mm256_movemask_epi8(_mm256_cmpeq_epi64(a, b));
        if (mask != -1) {
            // At least one lane differs. Find highest differing lane.
            // Lanes: [i-4]=bits0-7, [i-3]=bits8-15, [i-2]=bits16-23, [i-1]=bits24-31
            unsigned long idx;
            _BitScanReverse(&idx, (unsigned long)(~mask));
            uint32_t pos = i - 4 + (idx >> 3);
            return ap[pos] > bp[pos] ? 1 : -1;
        }
    }
    for (; i-- > 0;) {
        if (ap[i] != bp[i])
            return ap[i] > bp[i] ? 1 : -1;
    }
    return 0;
}

// ============================================================
// Normalize
// ============================================================

// Return the number of significant limbs (stripping leading zeros)
inline uint32_t mpn_normalize(const limb_t* ap, uint32_t n) {
    while (n > 0 && ap[n - 1] == 0) n--;
    return n;
}

// ============================================================
// Single-limb multiply
// ============================================================

// rp[0..n) = ap[0..n) * b, return carry limb
// 4x unrolled: multiplies are independent, carry chain is serial.
inline limb_t mpn_mul_1(limb_t* rp, const limb_t* ap, uint32_t n, limb_t b) {
    limb_t carry = 0;
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        limb_t hi0, hi1, hi2, hi3;
        limb_t lo0 = umul_hilo(ap[i],   b, &hi0);
        limb_t lo1 = umul_hilo(ap[i+1], b, &hi1);
        limb_t lo2 = umul_hilo(ap[i+2], b, &hi2);
        limb_t lo3 = umul_hilo(ap[i+3], b, &hi3);
        unsigned char c;
        c = _addcarry_u64(0, lo0, carry, (unsigned long long*)&rp[i]);
        carry = hi0 + c;
        c = _addcarry_u64(0, lo1, carry, (unsigned long long*)&rp[i+1]);
        carry = hi1 + c;
        c = _addcarry_u64(0, lo2, carry, (unsigned long long*)&rp[i+2]);
        carry = hi2 + c;
        c = _addcarry_u64(0, lo3, carry, (unsigned long long*)&rp[i+3]);
        carry = hi3 + c;
    }
    for (; i < n; i++) {
        limb_t hi;
        limb_t lo = umul_hilo(ap[i], b, &hi);
        unsigned char c = _addcarry_u64(0, lo, carry, (unsigned long long*)&rp[i]);
        carry = hi + c;
    }
    return carry;
}

extern "C" std::uint64_t zint_mpn_addmul_1_adx(std::uint64_t* rp, const std::uint64_t* ap, std::uint32_t n, std::uint64_t b);

// rp[0..n) += ap[0..n) * b, return carry limb
// Scalar for tiny n (avoids call overhead), ADX asm for n>=3 (3x faster).
inline limb_t mpn_addmul_1(limb_t* rp, const limb_t* ap, uint32_t n, limb_t b) {
    return (limb_t)zint_mpn_addmul_1_adx((std::uint64_t*)rp, (const std::uint64_t*)ap, n, (std::uint64_t)b);
}

extern "C" std::uint64_t zint_mpn_submul_1_adx(std::uint64_t* rp, const std::uint64_t* ap, std::uint32_t n, std::uint64_t b);

// rp[0..n) -= ap[0..n) * b, return borrow limb
// Scalar for tiny n, BMI2 asm for n>=3 (1.8x faster).
inline limb_t mpn_submul_1(limb_t* rp, const limb_t* ap, uint32_t n, limb_t b) {
    return (limb_t)zint_mpn_submul_1_adx((std::uint64_t*)rp, (const std::uint64_t*)ap, n, (std::uint64_t)b);
}

// ============================================================
// Fused mul_basecase (ASM)
// ============================================================

extern "C" void zint_mpn_mul_basecase_adx(std::uint64_t* rp, const std::uint64_t* ap, std::uint32_t an,
                                           const std::uint64_t* bp, std::uint32_t bn);

// ============================================================
// Single-limb division (Barrett / precomputed inverse)
// ============================================================

// Compute Barrett inverse for normalized divisor d (MSB set).
// Returns inv = floor((B^2 - 1) / d) - B where B = 2^64.
inline limb_t invert_limb(limb_t d) {
    assert(d >> 63);  // d must be normalized
    limb_t dummy;
    // floor(((B-1-d)*B + (B-1)) / d) = floor((B^2-1)/d) - B
    return udiv_128(~d, ~(limb_t)0, d, &dummy);
}

// Divide n1:n0 by d using precomputed inverse dinv (GMP algorithm).
// Precondition: n1 < d, d normalized (MSB set).
// Returns quotient, stores remainder in *rem.
ZINT_FORCEINLINE limb_t udiv_qrnnd_preinv(limb_t n1, limb_t n0, limb_t d,
                                            limb_t dinv, limb_t* rem) {
    limb_t q0, q1;
    q0 = umul_hilo(n1, dinv, &q1);
    q1 += n1 + 1;  // +1 is critical: ensures estimate is off by at most 1
    limb_t old_q0 = q0;
    q0 += n0;
    q1 += (q0 < old_q0);

    limb_t r = n0 - q1 * d;  // truncated 64-bit multiply
    if (r > q0) { q1--; r += d; }  // adjustment 1
    if (r >= d) { q1++; r -= d; }  // adjustment 2 (rare)

    *rem = r;
    return q1;
}

// Divide ap[0..n) by d. Store quotient in qp[0..n). Return remainder.
// Precondition: d > 0. qp may alias ap.
// Uses Barrett (precomputed inverse) for n >= 3, hardware div for tiny n.
inline limb_t mpn_divrem_1(limb_t* qp, const limb_t* ap, uint32_t n, limb_t d) {
    assert(n > 0 && d > 0);

    // Hardware div for tiny n (Barrett setup not amortized)
    if (n < 3) {
        limb_t rem = 0;
        for (uint32_t i = n; i-- > 0;)
            qp[i] = udiv_128(rem, ap[i], d, &rem);
        return rem;
    }

    unsigned cnt = clz64(d);

    if (cnt == 0) {
        // d is already normalized
        limb_t dinv = invert_limb(d);
        limb_t rem = 0;
        uint32_t i = n;
        for (; i >= 4; i -= 4) {
            qp[i-1] = udiv_qrnnd_preinv(rem, ap[i-1], d, dinv, &rem);
            qp[i-2] = udiv_qrnnd_preinv(rem, ap[i-2], d, dinv, &rem);
            qp[i-3] = udiv_qrnnd_preinv(rem, ap[i-3], d, dinv, &rem);
            qp[i-4] = udiv_qrnnd_preinv(rem, ap[i-4], d, dinv, &rem);
        }
        for (; i > 0; i--)
            qp[i-1] = udiv_qrnnd_preinv(rem, ap[i-1], d, dinv, &rem);
        return rem;
    }

    // Unnormalized: shift d left so MSB is set, adjust numerator on the fly
    limb_t d_norm = d << cnt;
    unsigned rcnt = 64 - cnt;
    limb_t dinv = invert_limb(d_norm);
    limb_t rem = ap[n - 1] >> rcnt;  // high bits of top limb (< d_norm guaranteed)

    uint32_t i = n - 1;
    for (; i >= 4; i -= 4) {
        limb_t nl;
        nl = (ap[i] << cnt) | (ap[i-1] >> rcnt);
        qp[i] = udiv_qrnnd_preinv(rem, nl, d_norm, dinv, &rem);
        nl = (ap[i-1] << cnt) | (ap[i-2] >> rcnt);
        qp[i-1] = udiv_qrnnd_preinv(rem, nl, d_norm, dinv, &rem);
        nl = (ap[i-2] << cnt) | (ap[i-3] >> rcnt);
        qp[i-2] = udiv_qrnnd_preinv(rem, nl, d_norm, dinv, &rem);
        nl = (ap[i-3] << cnt) | (ap[i-4] >> rcnt);
        qp[i-3] = udiv_qrnnd_preinv(rem, nl, d_norm, dinv, &rem);
    }
    for (; i > 0; i--) {
        limb_t nl = (ap[i] << cnt) | (ap[i-1] >> rcnt);
        qp[i] = udiv_qrnnd_preinv(rem, nl, d_norm, dinv, &rem);
    }
    qp[0] = udiv_qrnnd_preinv(rem, ap[0] << cnt, d_norm, dinv, &rem);

    return rem >> cnt;  // unnormalize remainder
}

// ============================================================
// Aligned memory allocation
// ============================================================

static constexpr size_t ALLOC_ALIGN = 32; // AVX2

inline limb_t* mpn_alloc(uint32_t n) {
    if (n == 0) return nullptr;
    size_t bytes = (size_t)n * sizeof(limb_t);
    // Round up to alignment
    bytes = (bytes + ALLOC_ALIGN - 1) & ~(ALLOC_ALIGN - 1);
#ifdef _MSC_VER
    return static_cast<limb_t*>(_aligned_malloc(bytes, ALLOC_ALIGN));
#else
    return static_cast<limb_t*>(std::aligned_alloc(ALLOC_ALIGN, bytes));
#endif
}

inline void mpn_free(limb_t* p) {
    if (!p) return;
#ifdef _MSC_VER
    _aligned_free(p);
#else
    std::free(p);
#endif
}

} // namespace zint
