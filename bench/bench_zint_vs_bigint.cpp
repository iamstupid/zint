// bench_zint_vs_bigint.cpp - Compare performance: legacy `bi` vs `zint`
//
// Build (MSVC, from parent repo root):
//   cl /std:c++17 /O2 /EHsc /arch:AVX2 zint\\bench\\bench_zint_vs_bigint.cpp /Fe:zint\\bench\\bench_zint_vs_bigint.exe

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bigint/mul.hpp"
#include "bigint/div.hpp"

#include "zint/mul.hpp"
#include "zint/div.hpp"
#include "zint/rng.hpp"

static inline std::int64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

static void fill_random_u64(std::uint64_t* buf, std::size_t n, zint::xoshiro256pp& rng) {
    for (std::size_t i = 0; i < n; ++i) buf[i] = rng.next_u64();
    if (n > 0) buf[n - 1] |= (1ULL << 63);
}

static double bench_mul_bi(std::uint32_t n, int iters) {
    zint::xoshiro256pp rng(123);
    std::vector<std::uint64_t> a(n), b(n), r(2 * (std::size_t)n);
    fill_random_u64(a.data(), n, rng);
    fill_random_u64(b.data(), n, rng);

    bi::mpn_mul((bi::limb_t*)r.data(), (const bi::limb_t*)a.data(), n,
                (const bi::limb_t*)b.data(), n);

    std::int64_t t0 = now_ns();
    for (int i = 0; i < iters; ++i) {
        bi::mpn_mul((bi::limb_t*)r.data(), (const bi::limb_t*)a.data(), n,
                    (const bi::limb_t*)b.data(), n);
    }
    std::int64_t t1 = now_ns();
    return double(t1 - t0) / iters;
}

static double bench_mul_zint(std::uint32_t n, int iters) {
    zint::xoshiro256pp rng(123);
    std::vector<std::uint64_t> a(n), b(n), r(2 * (std::size_t)n);
    fill_random_u64(a.data(), n, rng);
    fill_random_u64(b.data(), n, rng);

    zint::mpn_mul((zint::limb_t*)r.data(), (const zint::limb_t*)a.data(), n,
                  (const zint::limb_t*)b.data(), n);

    std::int64_t t0 = now_ns();
    for (int i = 0; i < iters; ++i) {
        zint::mpn_mul((zint::limb_t*)r.data(), (const zint::limb_t*)a.data(), n,
                      (const zint::limb_t*)b.data(), n);
    }
    std::int64_t t1 = now_ns();
    return double(t1 - t0) / iters;
}

static double bench_tdiv_bi(std::uint32_t n, int iters) {
    zint::xoshiro256pp rng(456);
    std::vector<std::uint64_t> num(2 * (std::size_t)n), den(n);
    fill_random_u64(num.data(), num.size(), rng);
    fill_random_u64(den.data(), den.size(), rng);
    den[n - 1] |= (1ULL << 63);

    std::vector<std::uint64_t> q(n + 1), r(n);

    bi::mpn_tdiv_qr((bi::limb_t*)q.data(), (bi::limb_t*)r.data(),
                    (const bi::limb_t*)num.data(), 2 * n,
                    (const bi::limb_t*)den.data(), n);

    std::int64_t t0 = now_ns();
    for (int i = 0; i < iters; ++i) {
        bi::mpn_tdiv_qr((bi::limb_t*)q.data(), (bi::limb_t*)r.data(),
                        (const bi::limb_t*)num.data(), 2 * n,
                        (const bi::limb_t*)den.data(), n);
    }
    std::int64_t t1 = now_ns();
    return double(t1 - t0) / iters;
}

static double bench_tdiv_zint(std::uint32_t n, int iters) {
    zint::xoshiro256pp rng(456);
    std::vector<std::uint64_t> num(2 * (std::size_t)n), den(n);
    fill_random_u64(num.data(), num.size(), rng);
    fill_random_u64(den.data(), den.size(), rng);
    den[n - 1] |= (1ULL << 63);

    std::vector<std::uint64_t> q(n + 1), r(n);

    zint::mpn_tdiv_qr((zint::limb_t*)q.data(), (zint::limb_t*)r.data(),
                      (const zint::limb_t*)num.data(), 2 * n,
                      (const zint::limb_t*)den.data(), n);

    std::int64_t t0 = now_ns();
    for (int i = 0; i < iters; ++i) {
        zint::mpn_tdiv_qr((zint::limb_t*)q.data(), (zint::limb_t*)r.data(),
                          (const zint::limb_t*)num.data(), 2 * n,
                          (const zint::limb_t*)den.data(), n);
    }
    std::int64_t t1 = now_ns();
    return double(t1 - t0) / iters;
}

struct Row {
    std::uint32_t n = 0;
    double bi_us = 0;
    double zint_us = 0;
    double ratio = 0;
};

static const char* arg_value(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return nullptr;
}

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

int main(int argc, char** argv) {
    const char* csv_mul = arg_value(argc, argv, "--csv-mul");
    const char* csv_div = arg_value(argc, argv, "--csv-div");

    struct Size { std::uint32_t n; int iters; };
    Size sizes[] = {
        {64,    5000},
        {128,   2000},
        {256,   1000},
        {512,    500},
        {1024,   200},
        {2048,   100},
        {4096,    50},
        {8192,    20},
        {16384,   10},
    };

    std::vector<Row> rows_mul;
    std::vector<Row> rows_div;
    rows_mul.reserve(std::size(sizes));
    rows_div.reserve(std::size(sizes));

    std::printf("\n=== mpn_mul: n x n (u64 limbs), time in microseconds ===\n\n");
    std::printf("%8s  %10s  %10s  %10s\n", "n", "bi", "zint", "zint/bi");
    std::printf("%8s  %10s  %10s  %10s\n", "------", "--------", "--------", "--------");
    for (auto s : sizes) {
        double bi_ns = bench_mul_bi(s.n, s.iters);
        double zi_ns = bench_mul_zint(s.n, s.iters);
        Row r;
        r.n = s.n;
        r.bi_us = bi_ns / 1000.0;
        r.zint_us = zi_ns / 1000.0;
        r.ratio = zi_ns / bi_ns;
        rows_mul.push_back(r);
        std::printf("%8u  %10.1f  %10.1f  %10.3f\n", r.n, r.bi_us, r.zint_us, r.ratio);
    }

    std::printf("\n=== mpn_tdiv_qr: (2n)/n, time in microseconds ===\n\n");
    std::printf("%8s  %10s  %10s  %10s\n", "n", "bi", "zint", "zint/bi");
    std::printf("%8s  %10s  %10s  %10s\n", "------", "--------", "--------", "--------");
    for (auto s : sizes) {
        int iters = s.iters / 4;
        if (iters < 3) iters = 3;
        double bi_ns = bench_tdiv_bi(s.n, iters);
        double zi_ns = bench_tdiv_zint(s.n, iters);
        Row r;
        r.n = s.n;
        r.bi_us = bi_ns / 1000.0;
        r.zint_us = zi_ns / 1000.0;
        r.ratio = zi_ns / bi_ns;
        rows_div.push_back(r);
        std::printf("%8u  %10.1f  %10.1f  %10.3f\n", r.n, r.bi_us, r.zint_us, r.ratio);
    }

    write_csv(csv_mul, rows_mul);
    write_csv(csv_div, rows_div);
    return 0;
}

