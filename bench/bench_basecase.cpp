// bench_basecase.cpp - Benchmark basecase mul/sqr kernels vs legacy bigint
//
// Measures:
// - mpn_mul_basecase: n x n
// - mpn_sqr_basecase: n^2
//
// CSV schema matches other comparisons: n,bi_us,zint_us,zint_over_bi

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "bigint/mul.hpp"
#include "zint/mul.hpp"
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
    if (n <= 8) return 150000;
    if (n <= 16) return 70000;
    if (n <= 32) return 30000;
    if (n <= 64) return 12000;
    return 6000;
}

struct Row {
    std::uint32_t n = 0;
    double bi_us = 0;
    double zint_us = 0;
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
        std::fprintf(f, "%u,%.3f,%.3f,%.6f\n", r.n, r.bi_us, r.zint_us, r.ratio);
    }
    std::fclose(f);
}

static volatile std::uint64_t sink_u64 = 0;

int main(int argc, char** argv) {
    const char* csv_mul = arg_value(argc, argv, "--csv-mul");
    const char* csv_sqr = arg_value(argc, argv, "--csv-sqr");

    std::uint32_t sizes[] = {2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64};

    zint::xoshiro256pp rng(20260211);

    std::vector<Row> rows_mul;
    std::vector<Row> rows_sqr;
    rows_mul.reserve(std::size(sizes));
    rows_sqr.reserve(std::size(sizes));

    for (std::uint32_t n : sizes) {
        int iters = iters_for_n(n);
        std::vector<std::uint64_t> a(n), b(n), r(2 * (std::size_t)n);
        fill_random_u64(a.data(), n, rng);
        fill_random_u64(b.data(), n, rng);

        double bi_ns = bench_best_ns(iters, [&]() {
            bi::mpn_mul_basecase((bi::limb_t*)r.data(), (const bi::limb_t*)a.data(), n,
                                 (const bi::limb_t*)b.data(), n);
        });
        sink_u64 ^= r[0];

        double zi_ns = bench_best_ns(iters, [&]() {
            zint::mpn_mul_basecase((zint::limb_t*)r.data(), (const zint::limb_t*)a.data(), n,
                                   (const zint::limb_t*)b.data(), n);
        });
        sink_u64 ^= r[0];

        rows_mul.push_back(Row{n, bi_ns / 1000.0, zi_ns / 1000.0, zi_ns / bi_ns});
    }

    for (std::uint32_t n : sizes) {
        int iters = iters_for_n(n);
        std::vector<std::uint64_t> a(n), r(2 * (std::size_t)n);
        fill_random_u64(a.data(), n, rng);

        double bi_ns = bench_best_ns(iters, [&]() { bi::mpn_sqr_basecase((bi::limb_t*)r.data(), (const bi::limb_t*)a.data(), n); });
        sink_u64 ^= r[0];

        double zi_ns = bench_best_ns(iters, [&]() { zint::mpn_sqr_basecase((zint::limb_t*)r.data(), (const zint::limb_t*)a.data(), n); });
        sink_u64 ^= r[0];

        rows_sqr.push_back(Row{n, bi_ns / 1000.0, zi_ns / 1000.0, zi_ns / bi_ns});
    }

    write_csv(csv_mul, rows_mul);
    write_csv(csv_sqr, rows_sqr);

    if (sink_u64 == 0x123456789ULL) std::printf("sink=%llu\n", (unsigned long long)sink_u64);
    return 0;
}

