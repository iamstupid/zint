#pragma once
// rng.hpp - fast small PRNGs used by tests/benchmarks
//
// xoshiro256++ reference: https://prng.di.unimi.it/xoshiro256plusplus.c

#include <cstdint>

namespace zint {

// splitmix64 for seeding xoshiro
struct splitmix64 {
    std::uint64_t s = 0;
    explicit splitmix64(std::uint64_t seed) : s(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

struct xoshiro256pp {
    std::uint64_t s[4] = {0, 0, 0, 0};

    xoshiro256pp() = default;
    explicit xoshiro256pp(std::uint64_t seed) { seed_with(seed); }

    static inline std::uint64_t rotl(std::uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    void seed_with(std::uint64_t seed) {
        splitmix64 sm(seed);
        s[0] = sm.next();
        s[1] = sm.next();
        s[2] = sm.next();
        s[3] = sm.next();
    }

    // next 64-bit output
    std::uint64_t next() {
        const std::uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const std::uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl(s[3], 45);

        return result;
    }

    std::uint32_t next_u32() { return static_cast<std::uint32_t>(next()); }
    std::uint64_t next_u64() { return next(); }
};

} // namespace zint

