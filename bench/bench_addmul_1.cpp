// bench_addmul_1.cpp - Benchmark mpn_addmul_1 scalar vs ADX-gated fast path
//
// Measures:
// - mpn_addmul_1 (scalar)
// - mpn_addmul_1_fast (uses ADX/BMI2 kernel when available)
//
// CSV schema: n,bi_us,zint_us,zint_over_bi

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "zint/mpn.hpp"
#include "zint/rng.hpp"

static inline std::int64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

template<class Fn>
static double bench_best_ns(int iters, Fn&& fn) {
    fn();
    std::int64_t best = (std::numeric_limits<std::int64_t>::max)();
    for (int t = 0; t < 7; ++t) {
        std::int64_t t0 = now_ns();
        for (int i = 0; i < iters; ++i) fn();
        std::int64_t t1 = now_ns();
        best = (std::min)(best, t1 - t0);
    }
    return double(best) / iters;
}

static const char* arg_value(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return nullptr;
}

static void fill_random_u64(std::uint64_t* buf, std::size_t n, zint::xoshiro256pp& rng) {
    for (std::size_t i = 0; i < n; ++i) buf[i] = rng.next_u64();
    if (n) buf[n - 1] |= (1ULL << 63);
}

static int iters_for_n(std::uint32_t n) {
    if (n <= 8) return 300000;
    if (n <= 16) return 150000;
    if (n <= 32) return 90000;
    if (n <= 64) return 45000;
    if (n <= 128) return 20000;
    return 12000;
}

struct Row {
    std::uint32_t n = 0;
    double base_us = 0;
    double fast_us = 0;
    double ratio = 0;
};

static void write_csv(const char* path, const std::vector<Row>& rows) {
    if (!path) return;
    std::FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::printf("warning: cannot open csv '%s'\n", path);
        return;
    }
    std::fprintf(f, "n,bi_us,zint_us,zint_over_bi\n");
    for (const auto& r : rows) {
        std::fprintf(f, "%u,%.3f,%.3f,%.6f\n", r.n, r.base_us, r.fast_us, r.ratio);
    }
    std::fclose(f);
}

static volatile std::uint64_t sink_u64 = 0;

int main(int argc, char** argv) {
    const char* csv = arg_value(argc, argv, "--csv");

    std::uint32_t sizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256};

    zint::xoshiro256pp rng(20260211);

    std::vector<Row> rows;
    rows.reserve(std::size(sizes));

    for (std::uint32_t n : sizes) {
        int iters = iters_for_n(n);
        std::vector<std::uint64_t> a(n), r0(n), r1(n), r2(n);
        fill_random_u64(a.data(), n, rng);
        fill_random_u64(r0.data(), n, rng);
        std::uint64_t b = rng.next_u64();

        r1 = r0;
        double base_ns = bench_best_ns(iters, [&]() {
            sink_u64 ^= (std::uint64_t)zint::mpn_addmul_1((zint::limb_t*)r1.data(), (const zint::limb_t*)a.data(), n, (zint::limb_t)b);
            sink_u64 ^= r1[0];
        });

        r2 = r0;
        double fast_ns = bench_best_ns(iters, [&]() {
            sink_u64 ^= (std::uint64_t)zint::mpn_addmul_1_fast((zint::limb_t*)r2.data(), (const zint::limb_t*)a.data(), n, (zint::limb_t)b);
            sink_u64 ^= r2[0];
        });

        rows.push_back(Row{n, base_ns / 1000.0, fast_ns / 1000.0, fast_ns / base_ns});
    }

    write_csv(csv, rows);

    if (sink_u64 == 0x123456789ULL) std::printf("sink=%llu\n", (unsigned long long)sink_u64);
    return 0;
}

