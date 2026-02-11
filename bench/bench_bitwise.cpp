// bench_bitwise.cpp - Bitwise performance benchmarks (mpn + bigint)
//
// Outputs:
// - mpn bitwise kernels: scalar baseline vs zint (AVX2) in the same CSV schema as other benches.
// - bigint bitwise operators: zint-only timing (includes sign-extension / two's-complement handling).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "zint/zint.hpp"
#include "zint/rng.hpp"

static inline std::int64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
}

template<class Fn>
static double bench_best_ns(int iters, Fn&& fn) {
    fn();
    std::int64_t best = (std::numeric_limits<std::int64_t>::max)();
    for (int t = 0; t < 5; ++t) {
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

static void fill_random(std::uint64_t* p, std::size_t n, zint::xoshiro256pp& rng) {
    for (std::size_t i = 0; i < n; ++i) p[i] = rng.next_u64();
    if (n) p[n - 1] |= (1ULL << 63);
}

static volatile std::uint64_t sink_u64 = 0;

// ------------------------------
// Scalar baselines (no intrinsics)
// ------------------------------

static void scalar_and(std::uint64_t* rp, const std::uint64_t* ap, const std::uint64_t* bp, std::uint32_t n) {
#if defined(_MSC_VER)
    __pragma(loop(no_vector))
#endif
    for (std::uint32_t i = 0; i < n; ++i) rp[i] = ap[i] & bp[i];
}

static void scalar_or(std::uint64_t* rp, const std::uint64_t* ap, const std::uint64_t* bp, std::uint32_t n) {
#if defined(_MSC_VER)
    __pragma(loop(no_vector))
#endif
    for (std::uint32_t i = 0; i < n; ++i) rp[i] = ap[i] | bp[i];
}

static void scalar_xor(std::uint64_t* rp, const std::uint64_t* ap, const std::uint64_t* bp, std::uint32_t n) {
#if defined(_MSC_VER)
    __pragma(loop(no_vector))
#endif
    for (std::uint32_t i = 0; i < n; ++i) rp[i] = ap[i] ^ bp[i];
}

static void scalar_not(std::uint64_t* rp, const std::uint64_t* ap, std::uint32_t n) {
#if defined(_MSC_VER)
    __pragma(loop(no_vector))
#endif
    for (std::uint32_t i = 0; i < n; ++i) rp[i] = ~ap[i];
}

struct RowCompare {
    std::uint32_t n = 0;
    double bi_us = 0;
    double zint_us = 0;
    double ratio = 0;
};

static void write_csv_compare(const char* path, const std::vector<RowCompare>& rows) {
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

struct RowBigint {
    std::uint32_t n = 0;
    const char* op = "";
    const char* scase = "";
    double us = 0;
};

static void write_csv_bigint(const char* path, const std::vector<RowBigint>& rows) {
    if (!path) return;
    std::FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::printf("warning: cannot open csv '%s'\n", path);
        return;
    }
    std::fprintf(f, "n,case,op,us\n");
    for (const auto& r : rows) {
        std::fprintf(f, "%u,%s,%s,%.3f\n", r.n, r.scase, r.op, r.us);
    }
    std::fclose(f);
}

static int iters_for_n(std::uint32_t n) {
    if (n <= 16) return 200000;
    if (n <= 64) return 60000;
    if (n <= 256) return 15000;
    if (n <= 1024) return 4000;
    if (n <= 4096) return 1000;
    return 250;
}

static zint::bigint make_bigint_from_limbs(const std::uint64_t* p, std::uint32_t n, bool neg) {
    return zint::bigint::from_limbs((const zint::limb_t*)p, n, neg);
}

static void bench_mpn_group(std::uint32_t n,
                            int iters,
                            std::uint64_t* r,
                            const std::uint64_t* a,
                            const std::uint64_t* b,
                            std::vector<RowCompare>& rows_and,
                            std::vector<RowCompare>& rows_or,
                            std::vector<RowCompare>& rows_xor,
                            std::vector<RowCompare>& rows_not)
{
    // and
    double s_and = bench_best_ns(iters, [&]() { scalar_and(r, a, b, n); });
    double z_and = bench_best_ns(iters, [&]() { zint::mpn_and_n((zint::limb_t*)r, (const zint::limb_t*)a, (const zint::limb_t*)b, n); });
    sink_u64 ^= r[0];
    rows_and.push_back({n, s_and / 1000.0, z_and / 1000.0, z_and / s_and});

    // or
    double s_or = bench_best_ns(iters, [&]() { scalar_or(r, a, b, n); });
    double z_or = bench_best_ns(iters, [&]() { zint::mpn_or_n((zint::limb_t*)r, (const zint::limb_t*)a, (const zint::limb_t*)b, n); });
    sink_u64 ^= r[0];
    rows_or.push_back({n, s_or / 1000.0, z_or / 1000.0, z_or / s_or});

    // xor
    double s_xor = bench_best_ns(iters, [&]() { scalar_xor(r, a, b, n); });
    double z_xor = bench_best_ns(iters, [&]() { zint::mpn_xor_n((zint::limb_t*)r, (const zint::limb_t*)a, (const zint::limb_t*)b, n); });
    sink_u64 ^= r[0];
    rows_xor.push_back({n, s_xor / 1000.0, z_xor / 1000.0, z_xor / s_xor});

    // not
    double s_not = bench_best_ns(iters, [&]() { scalar_not(r, a, n); });
    double z_not = bench_best_ns(iters, [&]() { zint::mpn_not_n((zint::limb_t*)r, (const zint::limb_t*)a, n); });
    sink_u64 ^= r[0];
    rows_not.push_back({n, s_not / 1000.0, z_not / 1000.0, z_not / s_not});
}

static void bench_bigint_group(std::uint32_t n,
                               int iters,
                               const zint::bigint& a,
                               const zint::bigint& b,
                               const char* scase,
                               std::vector<RowBigint>& rows)
{
    zint::bigint r = make_bigint_from_limbs(nullptr, 0, false);
    // Pre-grow capacity (avoid heap in the loop if possible) by assigning a big value once.
    {
        std::vector<std::uint64_t> tmp((std::size_t)n + 1);
        for (std::size_t i = 0; i < tmp.size(); ++i) tmp[i] = 0xFFFFFFFFFFFFFFFFULL;
        tmp.back() |= (1ULL << 63);
        r = make_bigint_from_limbs(tmp.data(), (std::uint32_t)tmp.size(), false);
    }

    auto bench_one = [&](const char* op, auto&& body) {
        double ns = bench_best_ns(iters, [&]() {
            r = a;
            body();
        });
        rows.push_back({n, op, scase, ns / 1000.0});
        sink_u64 ^= (std::uint64_t)r.abs_size();
    };

    bench_one("and", [&]() { r &= b; });
    bench_one("or", [&]() { r |= b; });
    bench_one("xor", [&]() { r ^= b; });
}

int main(int argc, char** argv) {
    const char* csv_and = arg_value(argc, argv, "--csv-mpn-and");
    const char* csv_or  = arg_value(argc, argv, "--csv-mpn-or");
    const char* csv_xor = arg_value(argc, argv, "--csv-mpn-xor");
    const char* csv_not = arg_value(argc, argv, "--csv-mpn-not");
    const char* csv_big = arg_value(argc, argv, "--csv-bigint");

    std::uint32_t sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    zint::xoshiro256pp rng(777);

    std::vector<RowCompare> rows_and;
    std::vector<RowCompare> rows_or;
    std::vector<RowCompare> rows_xor;
    std::vector<RowCompare> rows_not;
    rows_and.reserve(std::size(sizes));
    rows_or.reserve(std::size(sizes));
    rows_xor.reserve(std::size(sizes));
    rows_not.reserve(std::size(sizes));

    std::vector<RowBigint> rows_big;
    rows_big.reserve(std::size(sizes) * 6);

    for (std::uint32_t n : sizes) {
        int iters = iters_for_n(n);
        std::vector<std::uint64_t> a(n), b(n), r(n);
        fill_random(a.data(), n, rng);
        fill_random(b.data(), n, rng);

        bench_mpn_group(n, iters, r.data(), a.data(), b.data(),
                        rows_and, rows_or, rows_xor, rows_not);

        // Bigint bitwise operators (two cases: both non-negative, both negative)
        zint::bigint ap = make_bigint_from_limbs(a.data(), n, false);
        zint::bigint bp = make_bigint_from_limbs(b.data(), n, false);
        zint::bigint an = make_bigint_from_limbs(a.data(), n, true);
        zint::bigint bn = make_bigint_from_limbs(b.data(), n, true);

        int it_big = (iters > 2000) ? (iters / 10) : iters;
        bench_bigint_group(n, it_big, ap, bp, "pp", rows_big);
        bench_bigint_group(n, it_big, an, bn, "nn", rows_big);
    }

    write_csv_compare(csv_and, rows_and);
    write_csv_compare(csv_or, rows_or);
    write_csv_compare(csv_xor, rows_xor);
    write_csv_compare(csv_not, rows_not);
    write_csv_bigint(csv_big, rows_big);

    // Prevent whole-program DCE on some builds.
    if (sink_u64 == 0x123456789ULL) std::printf("sink=%llu\n", (unsigned long long)sink_u64);
    return 0;
}
