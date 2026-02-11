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
    int bases[] = {2, 3, 8, 16, 36};
    zint::xoshiro256pp rng(4);
    for (int base : bases) {
        for (int t = 0; t < 200; ++t) {
            zint::bigint a = random_bigint(rng, 64);
            std::string s = a.to_string(base);
            zint::bigint b = zint::bigint::from_string(s.c_str(), base);
            ZINT_ASSERT(b == a);
        }
    }
}

int main() {
    test_small_int64();
    test_modular_arithmetic();
    test_decimal_roundtrip_and_cache();
    test_non_decimal_roundtrip();
    std::printf("OK\n");
    return 0;
}

