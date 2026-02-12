// test_zint_correctness.cpp - Standalone correctness tests for zint
//
// Policy: must not compare against legacy bigint implementation.
// Uses arithmetic invariants and congruence checks modulo small primes.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "zint/zint.hpp"
#include "zint/rng.hpp"

static void fail(const char* file, int line, const char* expr) {
    std::printf("FAIL %s:%d: %s\n", file, line, expr);
    std::fflush(stdout);
    std::exit(1);
}

#define ZINT_ASSERT(x) do { if (!(x)) fail(__FILE__, __LINE__, #x); } while (0)

using u64 = std::uint64_t;
using u32 = std::uint32_t;

static u64 mod_mag_u64(const zint::limb_t* limbs, u32 n, u64 mod) {
    if (mod == 1) return 0;
    u64 r = 0;
    for (u32 i = n; i-- > 0;) {
        zint::limb_t rem = 0;
        (void)zint::udiv_128((zint::limb_t)r, (zint::limb_t)limbs[i], (zint::limb_t)mod, &rem);
        r = (u64)rem;
    }
    return r;
}

static u64 mod_bigint_u64(const zint::bigint& x, u64 mod) {
    if (mod == 1) return 0;
    u64 r = mod_mag_u64(x.limbs(), x.abs_size(), mod);
    if (x.is_negative() && r) r = mod - r;
    return r;
}

static u64 mod_from_decimal_string(const std::string& s, u64 mod) {
    if (mod == 1) return 0;
    std::size_t i = 0;
    bool neg = false;
    if (!s.empty() && (s[0] == '-' || s[0] == '+')) {
        neg = (s[0] == '-');
        i = 1;
    }
    u64 r = 0;
    for (; i < s.size(); ++i) {
        char c = s[i];
        if (c < '0' || c > '9') break;
        // mod values in tests are small (<= 1e9), so r*10 fits in u64.
        r = (r * 10 + (u64)(c - '0')) % mod;
    }
    if (neg && r) r = mod - r;
    return r;
}

static zint::bigint random_bigint(zint::xoshiro256pp& rng, u32 max_limbs) {
    u32 n = (max_limbs == 0) ? 0 : (rng.next_u32() % (max_limbs + 1));
    std::vector<zint::limb_t> v(n);
    for (u32 i = 0; i < n; ++i) v[i] = (zint::limb_t)rng.next_u64();
    if (n) {
        // ensure non-zero top limb (also sets a high bit so bit_length exercises paths)
        v[n - 1] |= (1ULL << 63);
    }
    bool neg = (rng.next_u64() & 1) != 0;
    return zint::bigint::from_limbs(v.data(), n, neg);
}

static void test_small_int64() {
    zint::xoshiro256pp rng(1);
    for (int t = 0; t < 20000; ++t) {
        // Keep values small enough that i64 ops won't overflow.
        std::int64_t a0 = (std::int64_t)(rng.next_u32() % 1000000000U);
        std::int64_t b0 = (std::int64_t)(rng.next_u32() % 1000000000U);
        if (rng.next_u64() & 1) a0 = -a0;
        if (rng.next_u64() & 1) b0 = -b0;
        if (b0 == 0) b0 = 1;

        zint::bigint a(a0), b(b0);

        zint::bigint s = a + b;
        ZINT_ASSERT(s == zint::bigint(a0 + b0));

        zint::bigint d = a - b;
        ZINT_ASSERT(d == zint::bigint(a0 - b0));

        zint::bigint m = a * b;
        ZINT_ASSERT(m == zint::bigint(a0 * b0));

        zint::bigint q = a / b;
        zint::bigint r = a % b;

        ZINT_ASSERT(q == zint::bigint(a0 / b0));
        ZINT_ASSERT(r == zint::bigint(a0 % b0));

        // Bitwise ops should match builtins for int64_t (two's complement semantics).
        zint::bigint aa = a & b;
        zint::bigint oo = a | b;
        zint::bigint xx = a ^ b;
        zint::bigint na = ~a;
        ZINT_ASSERT(aa == zint::bigint(a0 & b0));
        ZINT_ASSERT(oo == zint::bigint(a0 | b0));
        ZINT_ASSERT(xx == zint::bigint(a0 ^ b0));
        ZINT_ASSERT(na == zint::bigint(~a0));

        // to_string/from_string for small values
        std::string sa = a.to_string(10);
        ZINT_ASSERT(sa == std::to_string(a0));
        zint::bigint pa = zint::bigint::from_string(sa.c_str(), 10);
        ZINT_ASSERT(pa == a);
    }
}

static void test_modular_arithmetic() {
    constexpr u64 primes[] = {
        1000000007ULL, 1000000009ULL, 998244353ULL, 1000003ULL, 10000019ULL
    };

    zint::xoshiro256pp rng(2);

    for (int t = 0; t < 2000; ++t) {
        zint::bigint a = random_bigint(rng, 128);
        zint::bigint b = random_bigint(rng, 128);

        zint::bigint add = a + b;
        zint::bigint sub = a - b;
        zint::bigint mul = a * b;

        for (u64 p : primes) {
            u64 am = mod_bigint_u64(a, p);
            u64 bm = mod_bigint_u64(b, p);
            ZINT_ASSERT(mod_bigint_u64(add, p) == (am + bm) % p);
            ZINT_ASSERT(mod_bigint_u64(sub, p) == (am + p - bm) % p);
            ZINT_ASSERT(mod_bigint_u64(mul, p) == (am * bm) % p);
        }

        // Square fast path
        zint::bigint sq = a * a;
        for (u64 p : primes) {
            u64 am = mod_bigint_u64(a, p);
            ZINT_ASSERT(mod_bigint_u64(sq, p) == (am * am) % p);
        }

        // Division invariants (use a smaller divisor to keep runtime reasonable)
        zint::bigint d = random_bigint(rng, 32);
        if (d.is_zero()) d = zint::bigint(1);

        zint::bigint q = a / d;
        zint::bigint r = a % d;
        zint::bigint recomposed = q * d + r;
        ZINT_ASSERT(recomposed == a);

        if (!r.is_zero()) {
            ZINT_ASSERT(r.is_negative() == a.is_negative());
            ZINT_ASSERT(r.abs().compare_abs(d.abs()) < 0);
        } else {
            ZINT_ASSERT(!r.is_negative());
        }
    }
}

static void test_decimal_roundtrip_and_cache() {
    constexpr u64 primes[] = {
        1000000007ULL, 1000000009ULL, 998244353ULL, 1000003ULL
    };

    zint::xoshiro256pp rng(3);
    zint::bigint::radix_powers_cache cache; // default-constructed (base=0); should self-init to base 10.

    for (int t = 0; t < 400; ++t) {
        zint::bigint a = random_bigint(rng, 256);

        std::string s0 = a.to_string(10, nullptr);
        std::string s1 = a.to_string(10, &cache);
        ZINT_ASSERT(s0 == s1);

        for (u64 p : primes) {
            ZINT_ASSERT(mod_from_decimal_string(s0, p) == mod_bigint_u64(a, p));
        }

        zint::bigint b0 = zint::bigint::from_string(s0.c_str(), 10, nullptr);
        zint::bigint b1 = zint::bigint::from_string(s1.c_str(), 10, &cache);
        ZINT_ASSERT(b0 == a);
        ZINT_ASSERT(b1 == a);
    }
}

static void test_non_decimal_roundtrip() {
    // Test ALL bases 2-64 with small values
    zint::xoshiro256pp rng(4);
    for (int base = 2; base <= 64; ++base) {
        for (int t = 0; t < 50; ++t) {
            zint::bigint a = random_bigint(rng, 64);
            std::string s = a.to_string(base);
            zint::bigint b = zint::bigint::from_string(s.c_str(), base);
            ZINT_ASSERT(b == a);
        }
    }

    // Test power-of-2 bases with known values
    ZINT_ASSERT(zint::bigint(255LL).to_string(2) == "11111111");
    ZINT_ASSERT(zint::bigint(255LL).to_string(16) == "FF");
    ZINT_ASSERT(zint::bigint(255LL).to_string(8) == "377");
    ZINT_ASSERT(zint::bigint(255LL).to_string(4) == "3333");
    ZINT_ASSERT(zint::bigint(255LL).to_string(32) == "7V");
    ZINT_ASSERT(zint::bigint(255LL).to_string(64) == "3/");
    ZINT_ASSERT(zint::bigint(0LL).to_string(16) == "0");
    ZINT_ASSERT(zint::bigint(-1LL).to_string(16) == "-1");

    // Case-insensitive parsing for bases <= 36
    ZINT_ASSERT(zint::bigint::from_string("ff", 16) == zint::bigint(255LL));
    ZINT_ASSERT(zint::bigint::from_string("FF", 16) == zint::bigint(255LL));

    // Larger sizes: test D&C path for non-base-10
    int large_bases[] = {3, 5, 7, 10, 12, 16, 36, 64};
    for (int base : large_bases) {
        for (int size : {10, 50, 200, 1000}) {
            zint::bigint a = random_bigint(rng, (u32)size);
            std::string s = a.to_string(base);
            zint::bigint b = zint::bigint::from_string(s.c_str(), base);
            ZINT_ASSERT(b == a);
        }
    }

    // Roundtrip with shared cache
    zint::bigint::radix_powers_cache cache(16);
    for (int t = 0; t < 100; ++t) {
        zint::bigint a = random_bigint(rng, 128);
        std::string s = a.to_string(16, &cache);
        zint::bigint b = zint::bigint::from_string(s.c_str(), 16, &cache);
        ZINT_ASSERT(b == a);
    }
}

static std::vector<zint::limb_t> mod_2k_tc(const zint::bigint& x, u32 kbits) {
    u32 n = (kbits + 63) / 64;
    std::vector<zint::limb_t> v(n, 0);
    u32 m = x.abs_size();
    u32 copy_n = (m < n) ? m : n;
    for (u32 i = 0; i < copy_n; ++i) v[i] = x.limbs()[i];

    if (x.is_negative()) {
        for (u32 i = 0; i < n; ++i) v[i] = ~v[i];
        if (n) (void)zint::mpn_add_1(v.data(), v.data(), n, 1);
    }

    u32 tail = kbits & 63u;
    if (tail && n) v[n - 1] &= ((1ULL << tail) - 1);
    return v;
}

static void test_bitwise_algebra_and_mod2k() {
    zint::xoshiro256pp rng(5);
    constexpr u32 ks[] = {1, 7, 63, 64, 65, 127, 128, 129, 191, 192, 255, 256};

    for (int t = 0; t < 1500; ++t) {
        zint::bigint a = random_bigint(rng, 128);
        zint::bigint b = random_bigint(rng, 128);
        zint::bigint c = random_bigint(rng, 128);

        ZINT_ASSERT((a & b) == (b & a));
        ZINT_ASSERT((a | b) == (b | a));
        ZINT_ASSERT((a ^ b) == (b ^ a));

        ZINT_ASSERT((a & (b | c)) == ((a & b) | (a & c)));
        ZINT_ASSERT((a | (b & c)) == ((a | b) & (a | c)));
        ZINT_ASSERT(((a ^ b) ^ c) == (a ^ (b ^ c)));
        ZINT_ASSERT(((a & b) & c) == (a & (b & c)));
        ZINT_ASSERT(((a | b) | c) == (a | (b | c)));

        ZINT_ASSERT((a & (~a)).is_zero());
        ZINT_ASSERT((a | (~a)) == zint::bigint(-1));
        ZINT_ASSERT((a ^ (~a)) == zint::bigint(-1));
        ZINT_ASSERT((~a) == ((-a) - 1));

        // x + y == (x ^ y) + 2*(x & y)
        zint::bigint lhs = a + b;
        zint::bigint rhs = (a ^ b) + ((a & b) << 1);
        ZINT_ASSERT(lhs == rhs);

        // Compare low bits via mod 2^k.
        zint::bigint ab_and = a & b;
        zint::bigint ab_or = a | b;
        zint::bigint ab_xor = a ^ b;

        for (u32 k : ks) {
            auto am = mod_2k_tc(a, k);
            auto bm = mod_2k_tc(b, k);
            auto rm_and = mod_2k_tc(ab_and, k);
            auto rm_or = mod_2k_tc(ab_or, k);
            auto rm_xor = mod_2k_tc(ab_xor, k);

            u32 n = (k + 63) / 64;
            u32 tail = k & 63u;
            zint::limb_t mask = tail ? ((1ULL << tail) - 1) : ~(zint::limb_t)0;

            for (u32 i = 0; i < n; ++i) {
                zint::limb_t exp_and = am[i] & bm[i];
                zint::limb_t exp_or  = am[i] | bm[i];
                zint::limb_t exp_xor = am[i] ^ bm[i];
                if (i + 1 == n) {
                    exp_and &= mask;
                    exp_or  &= mask;
                    exp_xor &= mask;
                }
                ZINT_ASSERT(rm_and[i] == exp_and);
                ZINT_ASSERT(rm_or[i] == exp_or);
                ZINT_ASSERT(rm_xor[i] == exp_xor);
            }
        }
    }
}

static void test_mpn_addmul_1_default() {
    zint::xoshiro256pp rng(6);

    // Scalar reference for cross-checking ASM addmul_1
    auto addmul_1_ref = [](zint::limb_t* rp, const zint::limb_t* ap, u32 n, zint::limb_t b) -> zint::limb_t {
        zint::limb_t carry = 0;
        for (u32 i = 0; i < n; i++) {
            zint::limb_t hi;
            zint::limb_t lo = zint::umul_hilo(ap[i], b, &hi);
            unsigned char c = _addcarry_u64(0, lo, carry, (unsigned long long*)&lo);
            carry = hi + c;
            c = _addcarry_u64(0, lo, rp[i], (unsigned long long*)&rp[i]);
            carry += c;
        }
        return carry;
    };

    for (int t = 0; t < 5000; ++t) {
        u32 n = (rng.next_u32() % 256) + 1;
        zint::limb_t b = (zint::limb_t)rng.next_u64();

        std::vector<zint::limb_t> a(n), r0(n), r1(n), r2(n);
        for (u32 i = 0; i < n; ++i) {
            a[i] = (zint::limb_t)rng.next_u64();
            r0[i] = (zint::limb_t)rng.next_u64();
        }

        r1 = r0;
        r2 = r0;

        zint::limb_t c1 = addmul_1_ref(r1.data(), a.data(), n, b);
        zint::limb_t c2 = zint::mpn_addmul_1(r2.data(), a.data(), n, b);

        ZINT_ASSERT(c1 == c2);
        ZINT_ASSERT(std::memcmp(r1.data(), r2.data(), (size_t)n * sizeof(zint::limb_t)) == 0);
    }
}

static void test_mpn_addlsh_n() {
    zint::xoshiro256pp rng(6);
    u32 sizes[] = {1, 2, 3, 4, 5, 8, 16, 32};
    unsigned shifts[] = {1, 7, 13, 31, 63};

    for (u32 n : sizes) {
        for (unsigned sh : shifts) {
            for (int t = 0; t < 200; ++t) {
                std::vector<zint::limb_t> a(n), b(n), r(n + 1);
                for (u32 i = 0; i < n; ++i) {
                    a[i] = (zint::limb_t)rng.next_u64();
                    b[i] = (zint::limb_t)rng.next_u64();
                }
                a[n - 1] |= (1ULL << 63);
                b[n - 1] |= (1ULL << 63);

                zint::bigint A = zint::bigint::from_limbs(a.data(), n, false);
                zint::bigint B = zint::bigint::from_limbs(b.data(), n, false);
                zint::bigint ref = A + (B << sh);

                zint::limb_t carry = zint::mpn_addlsh_n(r.data(), a.data(), b.data(), n, sh);
                r[n] = carry;
                zint::bigint got = zint::bigint::from_limbs(r.data(), n + 1, false);
                ZINT_ASSERT(got == ref);

                // Aliasing: rp == ap should be supported.
                std::vector<zint::limb_t> a2 = a;
                std::vector<zint::limb_t> r2(n + 1);
                zint::limb_t carry2 = zint::mpn_addlsh_n(a2.data(), a2.data(), b.data(), n, sh);
                std::memcpy(r2.data(), a2.data(), (size_t)n * sizeof(zint::limb_t));
                r2[n] = carry2;
                zint::bigint got2 = zint::bigint::from_limbs(r2.data(), n + 1, false);
                ZINT_ASSERT(got2 == ref);
            }
        }
    }
}

int main() {
    test_small_int64();
    test_modular_arithmetic();
    test_decimal_roundtrip_and_cache();
    test_non_decimal_roundtrip();
    test_bitwise_algebra_and_mod2k();
    test_mpn_addmul_1_default();
    test_mpn_addlsh_n();
    std::printf("OK\n");
    return 0;
}
