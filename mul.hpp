#pragma once
// mul.hpp - Multiplication algorithms for arbitrary-precision integers
//
// Dispatch chain: basecase (O(n^2)) -> Karatsuba (O(n^1.585)) -> NTT (O(n log n))
// NTT bridge uses ntt/p50x4 engine via ntt::big_multiply_u64().

#include "mpn.hpp"
#include "scratch.hpp"
#include "tuning.hpp"
#include <cstring>
#include <algorithm>

// NTT bridge: include the NTT API for large multiplications
#include "ntt/api.hpp"

namespace zint {

// ============================================================
// Basecase multiplication (fused ASM schoolbook)
// ============================================================

// rp[0..an+bn) = ap[0..an) * bp[0..bn)
// Precondition: an >= bn > 0; rp does NOT alias ap or bp
ZINT_NOINLINE inline void mpn_mul_basecase(limb_t* rp, const limb_t* ap, uint32_t an,
                                            const limb_t* bp, uint32_t bn)
{
    zint_mpn_mul_basecase_adx((std::uint64_t*)rp, (const std::uint64_t*)ap, an,
                               (const std::uint64_t*)bp, bn);
}

// ============================================================
// Basecase squaring (exploits symmetry)
// ============================================================

// rp[0..2*n) = ap[0..n)^2
// Uses the identity: a^2 = sum_i(a[i]^2 * B^(2i)) + 2*sum_{i<j}(a[i]*a[j]*B^(i+j))
inline void mpn_sqr_basecase(limb_t* rp, const limb_t* ap, uint32_t n) {
    if (n == 1) {
        rp[0] = umul_hilo(ap[0], ap[0], &rp[1]);
        return;
    }

    // Compute off-diagonal products (upper triangle only)
    // rp = ap[0] * ap[1..n)
    rp[n] = mpn_mul_1(rp + 1, ap + 1, n - 1, ap[0]);
    rp[0] = 0; // will be filled by diagonal

    // Add remaining off-diagonal products
    if (n > 2) {
        for (uint32_t i = 1; i < n - 1; i++) {
            rp[i + n] = mpn_addmul_1(rp + 2 * i + 1, ap + i + 1, n - i - 1, ap[i]);
        }
    }
    rp[2 * n - 1] = 0;

    // Fused: double off-diagonal (lshift by 1) + add diagonal (ap[i]^2)
    // in a single pass, saving one full traversal of 2n-2 limbs.
    limb_t shift_carry = 0;
    unsigned char add_carry = 0;
    for (uint32_t i = 0; i < n; i++) {
        limb_t r0 = rp[2 * i];
        limb_t r1 = rp[2 * i + 1];
        limb_t d0 = (r0 << 1) | shift_carry;
        limb_t d1 = (r1 << 1) | (r0 >> 63);
        shift_carry = r1 >> 63;
        limb_t hi, lo;
        lo = umul_hilo(ap[i], ap[i], &hi);
        add_carry = _addcarry_u64(add_carry, d0, lo, (unsigned long long*)&rp[2 * i]);
        add_carry = _addcarry_u64(add_carry, d1, hi, (unsigned long long*)&rp[2 * i + 1]);
    }
}

// ============================================================
// Karatsuba multiplication
// ============================================================

// Scratch space needed: about 4*n limbs for the deepest recursion level
// (each level needs ~2n scratch, geometric series converges to ~4n)

// Internal Karatsuba: rp[0..2*n) = ap[0..n) * bp[0..n)
// scratch must have at least 4*n limbs available
inline void mpn_mul_karatsuba_n(limb_t* rp, const limb_t* ap, const limb_t* bp,
                                 uint32_t n, limb_t* scratch)
{
    if (n < KARATSUBA_THRESHOLD) {
        mpn_mul_basecase(rp, ap, n, bp, n);
        return;
    }

    uint32_t m = n / 2;        // low half size
    uint32_t h = n - m;        // high half size (h >= m, h == m+1 when n odd)

    const limb_t* a0 = ap;
    const limb_t* a1 = ap + m;
    const limb_t* b0 = bp;
    const limb_t* b1 = bp + m;

    // Step 1: z0 = a0 * b0 (stored in rp[0..2m))
    mpn_mul_karatsuba_n(rp, a0, b0, m, scratch);

    // Step 2: z2 = a1 * b1 (stored in rp[2m..2n))
    mpn_mul_karatsuba_n(rp + 2 * m, a1, b1, h, scratch);

    // Step 3: Compute |a0 - a1| and |b0 - b1|, track signs
    limb_t* t0 = scratch;
    limb_t* t1 = scratch + h;
    limb_t* t2 = scratch + 2 * h;

    bool neg_a, neg_b;

    // t0 = |a0 - a1| (h limbs)
    if (h > m) {
        if (a1[m] != 0) {
            neg_a = true;
        } else {
            neg_a = (mpn_cmp(a0, a1, m) < 0);
        }
    } else {
        neg_a = (mpn_cmp(a0, a1, m) < 0);
    }

    if (!neg_a) {
        if (h == m) {
            mpn_sub_n(t0, a0, a1, m);
        } else {
            limb_t borrow = mpn_sub_n(t0, a0, a1, m);
            t0[m] = 0 - a1[m] - borrow;
        }
    } else {
        if (h == m) {
            mpn_sub_n(t0, a1, a0, m);
        } else {
            mpn_copyi(t0, a1, h);
            mpn_sub(t0, t0, h, a0, m);
        }
    }

    // t1 = |b0 - b1| (h limbs)
    if (h > m) {
        if (b1[m] != 0) {
            neg_b = true;
        } else {
            neg_b = (mpn_cmp(b0, b1, m) < 0);
        }
    } else {
        neg_b = (mpn_cmp(b0, b1, m) < 0);
    }

    if (!neg_b) {
        if (h == m) {
            mpn_sub_n(t1, b0, b1, m);
        } else {
            limb_t borrow = mpn_sub_n(t1, b0, b1, m);
            t1[m] = 0 - b1[m] - borrow;
        }
    } else {
        if (h == m) {
            mpn_sub_n(t1, b1, b0, m);
        } else {
            mpn_copyi(t1, b1, h);
            mpn_sub(t1, t1, h, b0, m);
        }
    }

    // Step 4: t2 = |a0-a1| * |b0-b1| (in t2[0..2h))
    // Only check for zero; otherwise multiply at full size h.
    if ((t0[h - 1] | t0[0]) == 0 && mpn_normalize(t0, h) == 0) {
        mpn_zero(t2, 2 * h);
    } else if ((t1[h - 1] | t1[0]) == 0 && mpn_normalize(t1, h) == 0) {
        mpn_zero(t2, 2 * h);
    } else {
        mpn_mul_karatsuba_n(t2, t0, t1, h, t2 + 2 * h);
    }

    // Step 5: Recombination
    // z1 = z0 + z2 ± t2, where ± depends on sign:
    //   same signs (neg_a == neg_b) → subtract t2
    //   diff signs (neg_a != neg_b) → add t2
    //
    // rp currently: [z0 (2m limbs)] [z2 (2h limbs)]
    // Compute z1 in scratch[0..2h), then add to rp[m..].
    // t2 is at scratch[2h..4h).

    std::memcpy(scratch, rp, 2 * m * sizeof(limb_t));
    if (h > m) { scratch[2 * m] = 0; scratch[2 * m + 1] = 0; }
    limb_t z1_hi = mpn_add_n(scratch, scratch, rp + 2 * m, 2 * h);

    if (neg_a == neg_b) {
        limb_t borrow = mpn_sub_n(scratch, scratch, t2, 2 * h);
        z1_hi -= borrow;
    } else {
        limb_t c = mpn_add_n(scratch, scratch, t2, 2 * h);
        z1_hi += c;
    }

    // Add z1 to rp[m..2n)
    limb_t carry = mpn_add(rp + m, rp + m, 2 * n - m, scratch, 2 * h);
    carry += z1_hi;
    if (carry && m + 2 * h < 2 * n) {
        mpn_add_1(rp + m + 2 * h, rp + m + 2 * h, 2 * n - m - 2 * h, (limb_t)carry);
    }
}

// General unbalanced Karatsuba: an >= bn
inline void mpn_mul_karatsuba(limb_t* rp, const limb_t* ap, uint32_t an,
                               const limb_t* bp, uint32_t bn, limb_t* scratch)
{
    if (an == bn) {
        mpn_mul_karatsuba_n(rp, ap, bp, an, scratch);
        return;
    }

    // Unbalanced: slice 'a' into chunks of size bn and multiply each
    mpn_zero(rp, an + bn);

    uint32_t pos = 0;
    while (pos + bn <= an) {
        // Multiply ap[pos..pos+bn) * bp[0..bn)
        // Use scratch for product + recursive scratch
        limb_t* prod = scratch;
        limb_t* rec_scratch = scratch + 2 * bn;

        mpn_mul_karatsuba_n(prod, ap + pos, bp, bn, rec_scratch);

        // Add to result at rp[pos..]
        limb_t carry = mpn_add(rp + pos, rp + pos, an + bn - pos, prod, 2 * bn);
        (void)carry;
        pos += bn;
    }

    // Handle remaining partial block
    uint32_t rem = an - pos;
    if (rem > 0) {
        uint32_t prod_n = rem + bn;
        uint32_t larger = (rem >= bn) ? rem : bn;
        uint32_t smaller = (rem >= bn) ? bn : rem;
        limb_t* prod = scratch;
        limb_t* rec_scratch = scratch + prod_n;

        if (smaller < KARATSUBA_THRESHOLD) {
            if (rem >= bn)
                mpn_mul_basecase(prod, ap + pos, rem, bp, bn);
            else
                mpn_mul_basecase(prod, bp, bn, ap + pos, rem);
        } else {
            if (rem >= bn)
                mpn_mul_karatsuba(prod, ap + pos, rem, bp, bn, rec_scratch);
            else
                mpn_mul_karatsuba(prod, bp, bn, ap + pos, rem, rec_scratch);
        }

        limb_t carry = mpn_add(rp + pos, rp + pos, an + bn - pos, prod, prod_n);
        (void)carry;
    }
}

// ============================================================
// Karatsuba squaring (exploits a==b symmetry)
// ============================================================

// rp[0..2*n) = ap[0..n)^2
// Saves ~33% per level: one |a0-a1| instead of two, recursive sqr instead of mul.
// Recombination: z1 = z0 + z2 - |a0-a1|^2 (sign is always subtract).
inline void mpn_sqr_karatsuba_n(limb_t* rp, const limb_t* ap,
                                 uint32_t n, limb_t* scratch)
{
    if (n < SQR_KARATSUBA_THRESHOLD) {
        mpn_sqr_basecase(rp, ap, n);
        return;
    }

    uint32_t m = n / 2;
    uint32_t h = n - m;

    const limb_t* a0 = ap;
    const limb_t* a1 = ap + m;

    // Step 1: z0 = a0^2 (in rp[0..2m))
    mpn_sqr_karatsuba_n(rp, a0, m, scratch);

    // Step 2: z2 = a1^2 (in rp[2m..2n))
    mpn_sqr_karatsuba_n(rp + 2 * m, a1, h, scratch);

    // Step 3: Compute |a0 - a1| into t0 (h limbs)
    limb_t* t0 = scratch;
    limb_t* t2 = scratch + 2 * h;  // keep same offset as mul version

    bool neg_a;
    if (h > m) {
        neg_a = (a1[m] != 0) || (mpn_cmp(a0, a1, m) < 0);
    } else {
        neg_a = (mpn_cmp(a0, a1, m) < 0);
    }

    if (!neg_a) {
        if (h == m) {
            mpn_sub_n(t0, a0, a1, m);
        } else {
            limb_t borrow = mpn_sub_n(t0, a0, a1, m);
            t0[m] = 0 - a1[m] - borrow;
        }
    } else {
        if (h == m) {
            mpn_sub_n(t0, a1, a0, m);
        } else {
            mpn_copyi(t0, a1, h);
            mpn_sub(t0, t0, h, a0, m);
        }
    }

    // Step 4: t2 = |a0-a1|^2 (in t2[0..2h))
    if ((t0[h - 1] | t0[0]) == 0 && mpn_normalize(t0, h) == 0) {
        mpn_zero(t2, 2 * h);
    } else {
        mpn_sqr_karatsuba_n(t2, t0, h, t2 + 2 * h);
    }

    // Step 5: Recombination — z1 = z0 + z2 - diff^2 (always subtract)
    std::memcpy(scratch, rp, 2 * m * sizeof(limb_t));
    if (h > m) { scratch[2 * m] = 0; scratch[2 * m + 1] = 0; }
    limb_t z1_hi = mpn_add_n(scratch, scratch, rp + 2 * m, 2 * h);

    limb_t borrow = mpn_sub_n(scratch, scratch, t2, 2 * h);
    z1_hi -= borrow;

    // Add z1 to rp[m..2n)
    limb_t carry = mpn_add(rp + m, rp + m, 2 * n - m, scratch, 2 * h);
    carry += z1_hi;
    if (carry && m + 2 * h < 2 * n) {
        mpn_add_1(rp + m + 2 * h, rp + m + 2 * h, 2 * n - m - 2 * h, (limb_t)carry);
    }
}

// ============================================================
// NTT multiplication (bridge to ntt/p50x4)
// ============================================================

inline void mpn_mul_ntt(limb_t* rp, const limb_t* ap, uint32_t an,
                         const limb_t* bp, uint32_t bn)
{
    ntt::big_multiply_u64(rp, (ntt::idt)(an + bn), ap, (ntt::idt)an, bp, (ntt::idt)bn);
}

// ============================================================
// Top-level multiply dispatch
// ============================================================

// rp[0..an+bn) = ap[0..an) * bp[0..bn)
// Precondition: an >= bn > 0; rp does not alias ap or bp
inline void mpn_mul(limb_t* rp, const limb_t* ap, uint32_t an,
                     const limb_t* bp, uint32_t bn)
{
    assert(an >= bn && bn > 0);

    if (bn < KARATSUBA_THRESHOLD) {
        mpn_mul_basecase(rp, ap, an, bp, bn);
    } else if (bn >= NTT_BN_MIN && (uint64_t)an * bn >= NTT_AREA) {
        mpn_mul_ntt(rp, ap, an, bp, bn);
    } else {
        uint32_t scratch_n = 6 * an + 128;
        ScratchScope scope(scratch());
        limb_t* scratchp = scope.alloc<limb_t>(scratch_n, 32);
        mpn_mul_karatsuba(rp, ap, an, bp, bn, scratchp);
    }
}

// Squaring: rp[0..2n) = ap[0..n)^2
inline void mpn_sqr(limb_t* rp, const limb_t* ap, uint32_t n) {
    assert(n > 0);

    if (n < SQR_KARATSUBA_THRESHOLD) {
        mpn_sqr_basecase(rp, ap, n);
    } else if (n < SQR_NTT_THRESHOLD) {
        uint32_t scratch_n = 6 * n + 128;
        ScratchScope scope(scratch());
        limb_t* scratchp = scope.alloc<limb_t>(scratch_n, 32);
        mpn_sqr_karatsuba_n(rp, ap, n, scratchp);
    } else {
        mpn_mul_ntt(rp, ap, n, ap, n);
    }
}

} // namespace zint
