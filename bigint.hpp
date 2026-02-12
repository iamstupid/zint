#pragma once
// bigint.hpp - Arbitrary-precision integer with C++ operators
//
// GMP-compatible memory layout (alloc, size with sign bit, data pointer).
// Provides both mpz_t-like raw access and mpz_class-like operator overloading.

#include "mpn.hpp"
#include "mul.hpp"
#include "div.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>
#include <cmath>

namespace zint {

struct bigint {
    using limb_t = zint::limb_t;
    static constexpr int LIMB_BITS = zint::LIMB_BITS;

private:
    uint32_t alloc_ = 0;
    uint32_t size_  = 0;  // bit 31 = sign; bits 0-30 = abs(limb count)
    limb_t*  data_  = nullptr;

    static constexpr uint32_t SIGN_BIT = 0x80000000u;
    static constexpr uint32_t SIZE_MASK = 0x7FFFFFFFu;

    enum class bitwise_op { and_, or_, xor_ };

    // ---- Memory management ----

    void ensure_capacity(uint32_t n) {
        if (n <= alloc_) return;
        uint32_t new_alloc = alloc_ < 4 ? 4 : alloc_;
        while (new_alloc < n) new_alloc *= 2;
        limb_t* new_data = mpn_alloc(new_alloc);
        uint32_t old_n = abs_size();
        if (old_n > 0 && data_) {
            std::memcpy(new_data, data_, (size_t)old_n * sizeof(limb_t));
        }
        mpn_free(data_);
        data_ = new_data;
        alloc_ = new_alloc;
    }

    void set_size_sign(uint32_t n, bool negative) {
        size_ = negative ? (n | SIGN_BIT) : n;
    }

    // Strip leading zeros and fix zero sign
    void trim() {
        uint32_t n = mpn_normalize(data_, abs_size());
        if (n == 0)
            size_ = 0; // zero is always non-negative
        else
            set_size_sign(n, is_negative());
    }

public:
    // ---- Properties ----

    bool is_zero() const noexcept { return (size_ & SIZE_MASK) == 0; }
    bool is_negative() const noexcept { return (size_ & SIGN_BIT) != 0; }
    bool is_positive() const noexcept { return !is_zero() && !is_negative(); }
    uint32_t abs_size() const noexcept { return size_ & SIZE_MASK; }
    const limb_t* limbs() const noexcept { return data_; }
    limb_t* limbs_mut() noexcept { return data_; }

    int sign() const noexcept {
        if (is_zero()) return 0;
        return is_negative() ? -1 : 1;
    }

    // Number of bits in absolute value (0 for zero)
    uint32_t bit_length() const noexcept {
        uint32_t n = abs_size();
        if (n == 0) return 0;
        return (n - 1) * LIMB_BITS + (LIMB_BITS - clz64(data_[n - 1]));
    }

    // ---- Constructors / Destructor ----

    bigint() noexcept = default;

    bigint(long long val) {  // NOLINT: use long long for unambiguous implicit conversion from int
        if (val == 0) return;
        bool neg = val < 0;
        // Handle INT64_MIN carefully
        uint64_t abs_val = neg ? (uint64_t)(-(val + 1)) + 1u : (uint64_t)val;
        ensure_capacity(1);
        data_[0] = abs_val;
        set_size_sign(1, neg);
    }

    bigint(const bigint& o) {
        if (o.is_zero()) return;
        uint32_t n = o.abs_size();
        ensure_capacity(n);
        std::memcpy(data_, o.data_, (size_t)n * sizeof(limb_t));
        size_ = o.size_;
    }

    bigint(bigint&& o) noexcept
        : alloc_(o.alloc_), size_(o.size_), data_(o.data_)
    {
        o.alloc_ = 0;
        o.size_ = 0;
        o.data_ = nullptr;
    }

    ~bigint() {
        mpn_free(data_);
    }

    // ---- Assignment ----

    bigint& operator=(const bigint& o) {
        if (this != &o) {
            if (o.is_zero()) {
                size_ = 0;
            } else {
                uint32_t n = o.abs_size();
                ensure_capacity(n);
                std::memcpy(data_, o.data_, (size_t)n * sizeof(limb_t));
                size_ = o.size_;
            }
        }
        return *this;
    }

    bigint& operator=(bigint&& o) noexcept {
        if (this != &o) {
            mpn_free(data_);
            alloc_ = o.alloc_;
            size_ = o.size_;
            data_ = o.data_;
            o.alloc_ = 0;
            o.size_ = 0;
            o.data_ = nullptr;
        }
        return *this;
    }

    bigint& operator=(long long val) {
        if (val == 0) {
            size_ = 0;
            return *this;
        }
        bool neg = val < 0;
        uint64_t abs_val = neg ? (uint64_t)(-(val + 1)) + 1u : (uint64_t)val;
        ensure_capacity(1);
        data_[0] = abs_val;
        set_size_sign(1, neg);
        return *this;
    }

    // ---- Swap ----

    void swap(bigint& o) noexcept {
        std::swap(alloc_, o.alloc_);
        std::swap(size_, o.size_);
        std::swap(data_, o.data_);
    }

    // ---- Comparison ----

    // Compare absolute values only
    int compare_abs(const bigint& o) const noexcept {
        uint32_t an = abs_size(), bn = o.abs_size();
        if (an != bn) return an > bn ? 1 : -1;
        if (an == 0) return 0;
        return mpn_cmp(data_, o.data_, an);
    }

    // Full signed comparison
    int compare(const bigint& o) const noexcept {
        int sa = sign(), sb = o.sign();
        if (sa != sb) return sa > sb ? 1 : -1;
        if (sa == 0) return 0;
        // Both same sign
        int cmp = compare_abs(o);
        return sa < 0 ? -cmp : cmp;
    }

    int compare(long long val) const noexcept {
        bigint tmp(val);
        return compare(tmp);
    }

    bool operator==(const bigint& o) const noexcept { return compare(o) == 0; }
    bool operator!=(const bigint& o) const noexcept { return compare(o) != 0; }
    bool operator<(const bigint& o)  const noexcept { return compare(o) < 0; }
    bool operator<=(const bigint& o) const noexcept { return compare(o) <= 0; }
    bool operator>(const bigint& o)  const noexcept { return compare(o) > 0; }
    bool operator>=(const bigint& o) const noexcept { return compare(o) >= 0; }

    bool operator==(long long val) const noexcept { return compare(val) == 0; }
    bool operator!=(long long val) const noexcept { return compare(val) != 0; }
    bool operator<(long long val)  const noexcept { return compare(val) < 0; }
    bool operator<=(long long val) const noexcept { return compare(val) <= 0; }
    bool operator>(long long val)  const noexcept { return compare(val) > 0; }
    bool operator>=(long long val) const noexcept { return compare(val) >= 0; }

    // ---- Unary ----

    bigint operator-() const {
        bigint r(*this);
        r.negate();
        return r;
    }

    bigint& negate() noexcept {
        if (!is_zero())
            size_ ^= SIGN_BIT;
        return *this;
    }

    bigint abs() const {
        bigint r(*this);
        r.size_ &= SIZE_MASK;
        return r;
    }

    // ---- Addition / Subtraction ----

    // Add absolute values: r = |a| + |b|
    static void add_magnitudes(bigint& r, const bigint& a, const bigint& b) {
        uint32_t an = a.abs_size(), bn = b.abs_size();
        if (an < bn) { add_magnitudes(r, b, a); return; } // ensure an >= bn

        if (bn == 0) {
            if (&r != &a) r = a;
            r.size_ &= SIZE_MASK; // clear sign (caller sets it)
            return;
        }

        r.ensure_capacity(an + 1);
        limb_t carry = mpn_add(r.data_, a.data_, an, b.data_, bn);
        if (carry) {
            r.data_[an] = carry;
            an++;
        }
        r.set_size_sign(an, false); // caller sets sign
    }

    // Subtract absolute values: r = |a| - |b| (assumes |a| >= |b|)
    static void sub_magnitudes(bigint& r, const bigint& a, const bigint& b) {
        uint32_t an = a.abs_size(), bn = b.abs_size();
        assert(an >= bn);

        if (bn == 0) {
            if (&r != &a) r = a;
            r.size_ &= SIZE_MASK;
            return;
        }

        r.ensure_capacity(an);
        limb_t borrow = mpn_sub(r.data_, a.data_, an, b.data_, bn);
        (void)borrow; // should be 0 since |a| >= |b|
        assert(borrow == 0);
        r.set_size_sign(an, false);
        r.trim();
    }

    bigint& operator+=(const bigint& o) {
        if (o.is_zero()) return *this;
        if (is_zero()) { *this = o; return *this; }

        bool neg_a = is_negative();
        bool neg_b = o.is_negative();

        if (neg_a == neg_b) {
            // Same sign: add magnitudes, keep sign
            bigint tmp;
            add_magnitudes(tmp, *this, o);
            tmp.set_size_sign(tmp.abs_size(), neg_a);
            *this = std::move(tmp);
        } else {
            // Different sign: subtract smaller from larger
            int cmp = compare_abs(o);
            if (cmp == 0) {
                size_ = 0; // result is zero
            } else if (cmp > 0) {
                // |this| > |o|: result sign = sign(this)
                bigint tmp;
                sub_magnitudes(tmp, *this, o);
                tmp.set_size_sign(tmp.abs_size(), neg_a);
                *this = std::move(tmp);
            } else {
                // |this| < |o|: result sign = sign(o)
                bigint tmp;
                sub_magnitudes(tmp, o, *this);
                tmp.set_size_sign(tmp.abs_size(), neg_b);
                *this = std::move(tmp);
            }
        }
        return *this;
    }

    bigint& operator-=(const bigint& o) {
        if (o.is_zero()) return *this;
        if (is_zero()) { *this = o; negate(); return *this; }

        bool neg_a = is_negative();
        bool neg_b = o.is_negative();

        if (neg_a != neg_b) {
            // Different sign: add magnitudes
            bigint tmp;
            add_magnitudes(tmp, *this, o);
            tmp.set_size_sign(tmp.abs_size(), neg_a);
            *this = std::move(tmp);
        } else {
            // Same sign: subtract magnitudes
            int cmp = compare_abs(o);
            if (cmp == 0) {
                size_ = 0;
            } else if (cmp > 0) {
                bigint tmp;
                sub_magnitudes(tmp, *this, o);
                tmp.set_size_sign(tmp.abs_size(), neg_a);
                *this = std::move(tmp);
            } else {
                bigint tmp;
                sub_magnitudes(tmp, o, *this);
                tmp.set_size_sign(tmp.abs_size(), !neg_a); // flip sign
                *this = std::move(tmp);
            }
        }
        return *this;
    }

    bigint operator+(const bigint& o) const { bigint r(*this); r += o; return r; }
    bigint operator-(const bigint& o) const { bigint r(*this); r -= o; return r; }

    bigint& operator+=(long long val) { bigint tmp(val); *this += tmp; return *this; }
    bigint& operator-=(long long val) { bigint tmp(val); *this -= tmp; return *this; }
    bigint operator+(long long val) const { bigint r(*this); r += val; return r; }
    bigint operator-(long long val) const { bigint r(*this); r -= val; return r; }

    // ---- Shift operations ----

    bigint& operator<<=(unsigned cnt) {
        if (is_zero() || cnt == 0) return *this;

        uint32_t limb_shift = cnt / LIMB_BITS;
        unsigned bit_shift = cnt % LIMB_BITS;
        uint32_t old_n = abs_size();
        bool neg = is_negative();

        uint32_t new_n = old_n + limb_shift + (bit_shift > 0 ? 1 : 0);
        ensure_capacity(new_n);

        if (bit_shift > 0) {
            // Bit shift first (processes high-to-low, safe for overlapping forward)
            limb_t carry = mpn_lshift(data_ + limb_shift, data_, old_n, bit_shift);
            if (carry) {
                data_[old_n + limb_shift] = carry;
            } else {
                new_n--;
            }
        } else {
            // Just move limbs up
            mpn_copyd(data_ + limb_shift, data_, old_n);
        }

        // Zero the low limbs
        mpn_zero(data_, limb_shift);
        set_size_sign(new_n, neg);
        return *this;
    }

    bigint& operator>>=(unsigned cnt) {
        if (is_zero() || cnt == 0) return *this;

        uint32_t limb_shift = cnt / LIMB_BITS;
        unsigned bit_shift = cnt % LIMB_BITS;
        uint32_t old_n = abs_size();
        bool neg = is_negative();

        if (limb_shift >= old_n) {
            size_ = 0;
            return *this;
        }

        uint32_t new_n = old_n - limb_shift;

        if (bit_shift > 0) {
            mpn_rshift(data_, data_ + limb_shift, new_n, bit_shift);
            if (data_[new_n - 1] == 0) new_n--;
        } else {
            mpn_copyi(data_, data_ + limb_shift, new_n);
        }

        if (new_n == 0)
            size_ = 0;
        else
            set_size_sign(new_n, neg);
        return *this;
    }

    bigint operator<<(unsigned cnt) const { bigint r(*this); r <<= cnt; return r; }
    bigint operator>>(unsigned cnt) const { bigint r(*this); r >>= cnt; return r; }

    // ---- Bitwise operations (infinite two's complement semantics) ----

    bigint operator~() const {
        // For infinite two's complement: ~x == -x - 1
        bigint r(*this);
        r.negate();
        r -= 1;
        return r;
    }

    bigint& operator&=(const bigint& o) {
        return bitwise_assign(o, bitwise_op::and_);
    }

    bigint& operator|=(const bigint& o) {
        return bitwise_assign(o, bitwise_op::or_);
    }

    bigint& operator^=(const bigint& o) {
        return bitwise_assign(o, bitwise_op::xor_);
    }

    bigint operator&(const bigint& o) const { return bitwise_binary_op(*this, o, bitwise_op::and_); }
    bigint operator|(const bigint& o) const { return bitwise_binary_op(*this, o, bitwise_op::or_); }
    bigint operator^(const bigint& o) const { return bitwise_binary_op(*this, o, bitwise_op::xor_); }

    // ---- Multiply by single limb (internal, used by string conversion) ----

    bigint& mul_limb(limb_t b) {
        if (is_zero() || b == 0) {
            size_ = 0;
            return *this;
        }
        if (b == 1) return *this;

        uint32_t n = abs_size();
        bool neg = is_negative();
        ensure_capacity(n + 1);
        limb_t carry = mpn_mul_1(data_, data_, n, b);
        if (carry) {
            data_[n] = carry;
            n++;
        }
        set_size_sign(n, neg);
        return *this;
    }

    // ---- Divide by single limb, return remainder (internal) ----

    limb_t div_limb(limb_t d) {
        assert(d > 0);
        if (is_zero()) return 0;

        uint32_t n = abs_size();
        limb_t rem = mpn_divrem_1(data_, data_, n, d);
        trim();
        return rem;
    }

    // ---- Add single limb to magnitude (internal) ----

    void add_limb(limb_t b) {
        if (b == 0) return;
        if (is_zero()) {
            ensure_capacity(1);
            data_[0] = b;
            set_size_sign(1, false);
            return;
        }
        uint32_t n = abs_size();
        ensure_capacity(n + 1);
        limb_t carry = mpn_add_1v(data_, data_, n, b);
        if (carry) {
            data_[n] = carry;
            n++;
        }
        set_size_sign(n, is_negative());
    }

    // ---- Radix conversion utilities ----

    static constexpr const char* RADIX_ALPHABET =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/";

    // Reverse mapping: char -> digit value, or -1 for invalid.
    // Bases <= 36: case-insensitive (both 'a' and 'A' map to 10).
    // Bases > 36: case-sensitive ('A'=10..35, 'a'=36..61, '+'=62, '/'=63).
    static int digit_value(char c, uint32_t base) {
        int v;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'A' && c <= 'Z') v = c - 'A' + 10;
        else if (c >= 'a' && c <= 'z') {
            v = (base <= 36) ? (c - 'a' + 10) : (c - 'a' + 36);
        }
        else if (c == '+') v = 62;
        else if (c == '/') v = 63;
        else return -1;
        return (v < (int)base) ? v : -1;
    }

    // Returns log2(base) if base is a power of 2, else 0.
    static unsigned pow2_shift(uint32_t base) {
        if (base < 2 || (base & (base - 1)) != 0) return 0;
        unsigned s = 0;
        uint32_t v = base;
        while (v > 1) { v >>= 1; s++; }
        return s;
    }

    // Max k where base^k fits in u64.
    static uint32_t compute_chunk_k(uint32_t base) {
        uint32_t k = 0;
        limb_t pow = 1;
        while (pow <= UINT64_MAX / base) {
            pow *= base;
            k++;
        }
        return k;
    }

    // Compute base^k as u64 (must fit).
    static limb_t compute_pow(uint32_t base, uint32_t k) {
        limb_t pow = 1;
        for (uint32_t i = 0; i < k; i++) pow *= base;
        return pow;
    }

    // Max t where base^t <= max_lut_size.
    static uint32_t compute_lut_t(uint32_t base, uint32_t max_lut_size) {
        uint32_t t = 0;
        uint32_t pow = 1;
        while (pow <= max_lut_size / base) {
            pow *= base;
            t++;
        }
        return t;
    }

    // Upper bound on number of base-b digits for a number with 'bits' bits.
    static uint32_t est_radix_digits(uint32_t bits, uint32_t base) {
        if (bits == 0) return 1;
        return (uint32_t)std::ceil((double)bits / std::log2((double)base)) + 1;
    }

    // ---- Radix power cache ----
    //
    // Stores chunk_pow^(2^k) lazily, where chunk_pow = base^chunk_k
    // and chunk_k = max digits fitting in one u64 limb.
    // Used by D&C radix conversion for any base 2-64.
    // Not thread-safe if shared across threads.
    struct radix_powers_cache {
        struct entry {
            limb_t* data = nullptr;      // chunk_pow^(2^k) = base^(chunk_k * 2^k)
            uint32_t size = 0;
            limb_t* d_norm = nullptr;    // normalized: data << shift
            limb_t* inv = nullptr;       // Newton reciprocal of d_norm
            unsigned shift = 0;
        };

        uint32_t radix = 0;              // user base (2..64)
        uint32_t chunk_k = 0;            // digits per machine-word chunk
        limb_t   chunk_pow = 0;          // base^chunk_k
        uint32_t lut_t_ = 0;            // LUT width (digits per LUT entry)
        uint32_t lut_pow = 0;           // base^lut_t_
        char*    to_str_lut = nullptr;   // lut_pow * lut_t_ chars
        std::vector<entry> pow2k{};      // pow2k[k] = chunk_pow^(2^k)

        radix_powers_cache() = default;
        explicit radix_powers_cache(uint32_t base) { reset(base); }

        radix_powers_cache(const radix_powers_cache&) = delete;
        radix_powers_cache& operator=(const radix_powers_cache&) = delete;

        radix_powers_cache(radix_powers_cache&& o) noexcept
            : radix(o.radix), chunk_k(o.chunk_k), chunk_pow(o.chunk_pow),
              lut_t_(o.lut_t_), lut_pow(o.lut_pow), to_str_lut(o.to_str_lut),
              pow2k(std::move(o.pow2k))
        {
            o.radix = 0;
            o.to_str_lut = nullptr;
            o.pow2k.clear();
        }
        radix_powers_cache& operator=(radix_powers_cache&& o) noexcept {
            if (this != &o) {
                clear();
                radix = o.radix; chunk_k = o.chunk_k; chunk_pow = o.chunk_pow;
                lut_t_ = o.lut_t_; lut_pow = o.lut_pow;
                to_str_lut = o.to_str_lut;
                pow2k = std::move(o.pow2k);
                o.radix = 0;
                o.to_str_lut = nullptr;
                o.pow2k.clear();
            }
            return *this;
        }

        ~radix_powers_cache() { clear(); }

        void clear() noexcept {
            for (auto& e : pow2k) {
                mpn_free(e.data);
                mpn_free(e.d_norm);
                mpn_free(e.inv);
            }
            pow2k.clear();
            delete[] to_str_lut;
            to_str_lut = nullptr;
            radix = 0;
        }

        void reset(uint32_t base) {
            if (radix == base && !pow2k.empty()) return;
            clear();
            radix = base;
            if (base < 2) return;

            chunk_k = compute_chunk_k(base);
            chunk_pow = compute_pow(base, chunk_k);
            lut_t_ = compute_lut_t(base, RADIX_LUT_MAX_SIZE);
            lut_pow = (uint32_t)compute_pow(base, lut_t_);

            // Build to_string LUT
            to_str_lut = new char[(size_t)lut_pow * lut_t_];
            for (uint32_t v = 0; v < lut_pow; v++) {
                char* dst = to_str_lut + (size_t)v * lut_t_;
                uint32_t tmp = v;
                for (int j = (int)lut_t_ - 1; j >= 0; j--) {
                    dst[j] = RADIX_ALPHABET[tmp % base];
                    tmp /= base;
                }
            }

            // pow2k[0] = chunk_pow (single limb)
            pow2k.push_back({mpn_alloc(1), 1, nullptr, nullptr, 0});
            pow2k.back().data[0] = chunk_pow;
        }

        const entry& get_pow2k(uint32_t k) {
            assert(radix >= 2);
            while (k >= (uint32_t)pow2k.size()) {
                const entry& prev = pow2k.back();
                uint32_t rn = 2 * prev.size;
                limb_t* rp = mpn_alloc(rn);
                mpn_sqr(rp, prev.data, prev.size);
                rn = mpn_normalize(rp, rn);
                pow2k.push_back({rp, rn, nullptr, nullptr, 0});
            }
            return pow2k[k];
        }

        void ensure_reciprocal(uint32_t k) {
            assert(k < (uint32_t)pow2k.size());
            auto& e = pow2k[k];
            if (e.inv) return;
            if (e.size < DIV_DC_THRESHOLD) return;

            e.shift = clz64(e.data[e.size - 1]);
            e.d_norm = mpn_alloc(e.size);
            if (e.shift > 0)
                mpn_lshift(e.d_norm, e.data, e.size, e.shift);
            else
                std::memcpy(e.d_norm, e.data, e.size * sizeof(limb_t));

            e.inv = mpn_alloc(e.size);
            mpn_newton_invert(e.inv, e.d_norm, e.size);
        }
    };

    // ---- String conversion ----
    // All non-power-of-2 bases use D&C O(M(n) log n).
    // Power-of-2 bases (2,4,8,16,32,64) use O(n) bit extraction.

    std::string to_string(int base = 10, radix_powers_cache* pow_cache = nullptr) const {
        assert(base >= 2 && base <= 64);
        if (is_zero()) return "0";

        unsigned shift = pow2_shift((uint32_t)base);
        if (shift) return pow2_to_string(*this, shift);

        // General D&C path for all non-power-of-2 bases
        std::string result;
        if (is_negative()) result.push_back('-');

        bigint tmp = this->abs();
        uint32_t est = est_radix_digits(bit_length(), (uint32_t)base);

        radix_powers_cache local;
        if (!pow_cache) {
            local.reset((uint32_t)base);
            pow_cache = &local;
        } else {
            if (pow_cache->radix == 0) pow_cache->reset((uint32_t)base);
            else assert(pow_cache->radix == (uint32_t)base);
        }
        dc_to_radix(result, tmp, est, 0, *pow_cache);
        return result;
    }

    static bigint from_string(const char* s, int base = 10, radix_powers_cache* pow_cache = nullptr) {
        assert(base >= 2 && base <= 64);
        if (!s || !*s) return bigint();

        bool neg = false;
        if (*s == '-') { neg = true; s++; }
        else if (*s == '+' && base <= 62) { s++; }

        // Skip leading zeros
        size_t len = std::strlen(s);
        while (len > 1 && *s == '0') { s++; len--; }

        unsigned shift = pow2_shift((uint32_t)base);
        if (shift) return pow2_from_string(s, (uint32_t)len, shift, neg);

        // General D&C path
        radix_powers_cache local;
        if (!pow_cache) {
            local.reset((uint32_t)base);
            pow_cache = &local;
        } else {
            if (pow_cache->radix == 0) pow_cache->reset((uint32_t)base);
            else assert(pow_cache->radix == (uint32_t)base);
        }

        bigint result = dc_from_radix(s, (uint32_t)len, *pow_cache);
        if (neg && !result.is_zero()) result.negate();
        return result;
    }

    // Construct from raw limbs (magnitude), little-endian.
    static bigint from_limbs(const limb_t* p, uint32_t n, bool neg = false) {
        bigint r;
        n = mpn_normalize(p, n);
        if (n == 0) return r;
        r.ensure_capacity(n);
        std::memcpy(r.data_, p, (size_t)n * sizeof(limb_t));
        r.set_size_sign(n, neg);
        return r;
    }

    // Construct from string (convenience)
    explicit bigint(const char* s, int base = 10) : bigint(from_string(s, base)) {}
    explicit bigint(const std::string& s, int base = 10) : bigint(from_string(s.c_str(), base)) {}

    // ---- Stream operators ----

    friend std::ostream& operator<<(std::ostream& os, const bigint& x) {
        return os << x.to_string();
    }

private:
    bigint& bitwise_assign(const bigint& o, bitwise_op op) {
        // Fast path: both operands non-negative => no sign extension needed.
        if (!is_negative() && !o.is_negative()) {
            uint32_t an = abs_size();
            uint32_t bn = o.abs_size();

            if (op == bitwise_op::and_) {
                uint32_t rn = (an < bn) ? an : bn;
                if (rn == 0) {
                    size_ = 0;
                    return *this;
                }
                mpn_and_n(data_, data_, o.data_, rn);
                set_size_sign(rn, false);
                trim();
                return *this;
            }

            if (an == 0) {
                *this = o;
                return *this;
            }
            if (bn == 0) {
                return *this;
            }

            uint32_t min_n = (an < bn) ? an : bn;
            uint32_t rn = (an > bn) ? an : bn;
            ensure_capacity(rn);

            if (op == bitwise_op::or_) mpn_or_n(data_, data_, o.data_, min_n);
            else mpn_xor_n(data_, data_, o.data_, min_n);

            if (bn > an) mpn_copyi(data_ + min_n, o.data_ + min_n, bn - min_n);
            set_size_sign(rn, false);
            trim();
            return *this;
        }

        uint32_t n = (std::max)(abs_size(), o.abs_size()) + 1;
        ScratchScope scope(scratch());
        limb_t* atc = scope.alloc<limb_t>(n, 32);
        limb_t* btc = scope.alloc<limb_t>(n, 32);
        limb_t* rtc = scope.alloc<limb_t>(n, 32);

        to_twos_complement(atc, n, *this);
        to_twos_complement(btc, n, o);

        switch (op) {
        case bitwise_op::and_: mpn_and_n(rtc, atc, btc, n); break;
        case bitwise_op::or_:  mpn_or_n(rtc, atc, btc, n); break;
        case bitwise_op::xor_: mpn_xor_n(rtc, atc, btc, n); break;
        }

        bool neg = (rtc[n - 1] >> (LIMB_BITS - 1)) != 0;
        if (!neg) {
            uint32_t m = mpn_normalize(rtc, n);
            if (m == 0) {
                size_ = 0;
                return *this;
            }
            ensure_capacity(m);
            std::memcpy(data_, rtc, (size_t)m * sizeof(limb_t));
            set_size_sign(m, false);
            return *this;
        }

        limb_t* tmp = scope.alloc<limb_t>(n, 32);
        mpn_not_n(tmp, rtc, n);
        (void)mpn_add_1v(tmp, tmp, n, 1);
        uint32_t m = mpn_normalize(tmp, n);
        if (m == 0) {
            size_ = 0;
            return *this;
        }
        ensure_capacity(m);
        std::memcpy(data_, tmp, (size_t)m * sizeof(limb_t));
        set_size_sign(m, true);
        return *this;
    }

    static void to_twos_complement(limb_t* rp, uint32_t n, const bigint& x) {
        assert(n > 0);
        mpn_zero(rp, n);
        uint32_t m = x.abs_size();
        if (m > 0) std::memcpy(rp, x.data_, (size_t)m * sizeof(limb_t));
        if (!x.is_negative()) return;

        // Two's complement: ~mag + 1 (within n limbs).
        mpn_not_n(rp, rp, n);
        (void)mpn_add_1v(rp, rp, n, 1);
    }

    static bigint from_twos_complement(const limb_t* ap, uint32_t n) {
        assert(n > 0);
        bool neg = (ap[n - 1] >> (LIMB_BITS - 1)) != 0;
        if (!neg) {
            uint32_t m = mpn_normalize(ap, n);
            return from_limbs(ap, m, false);
        }

        ScratchScope scope(scratch());
        limb_t* tmp = scope.alloc<limb_t>(n, 32);
        mpn_not_n(tmp, ap, n);
        (void)mpn_add_1v(tmp, tmp, n, 1);
        uint32_t m = mpn_normalize(tmp, n);
        return from_limbs(tmp, m, true);
    }

    static bigint bitwise_binary_op(const bigint& a, const bigint& b, bitwise_op op) {
        // Fast path: both operands non-negative => no sign extension needed.
        if (!a.is_negative() && !b.is_negative()) {
            uint32_t an = a.abs_size();
            uint32_t bn = b.abs_size();

            if (op == bitwise_op::and_) {
                uint32_t rn = (an < bn) ? an : bn;
                if (rn == 0) return bigint();
                bigint r;
                r.ensure_capacity(rn);
                mpn_and_n(r.data_, a.data_, b.data_, rn);
                r.set_size_sign(rn, false);
                r.trim();
                return r;
            }

            if (an == 0) return b;
            if (bn == 0) return a;

            uint32_t min_n = (an < bn) ? an : bn;
            uint32_t rn = (an > bn) ? an : bn;
            bigint r;
            r.ensure_capacity(rn);

            if (op == bitwise_op::or_) mpn_or_n(r.data_, a.data_, b.data_, min_n);
            else mpn_xor_n(r.data_, a.data_, b.data_, min_n);

            if (an > min_n) mpn_copyi(r.data_ + min_n, a.data_ + min_n, an - min_n);
            else if (bn > min_n) mpn_copyi(r.data_ + min_n, b.data_ + min_n, bn - min_n);

            r.set_size_sign(rn, false);
            r.trim();
            return r;
        }

        uint32_t n = (std::max)(a.abs_size(), b.abs_size()) + 1;
        ScratchScope scope(scratch());
        limb_t* atc = scope.alloc<limb_t>(n, 32);
        limb_t* btc = scope.alloc<limb_t>(n, 32);
        limb_t* rtc = scope.alloc<limb_t>(n, 32);

        to_twos_complement(atc, n, a);
        to_twos_complement(btc, n, b);

        switch (op) {
        case bitwise_op::and_: mpn_and_n(rtc, atc, btc, n); break;
        case bitwise_op::or_:  mpn_or_n(rtc, atc, btc, n); break;
        case bitwise_op::xor_: mpn_xor_n(rtc, atc, btc, n); break;
        }

        return from_twos_complement(rtc, n);
    }

    // Construct non-negative bigint from raw limb array
    static bigint from_limbs_unsigned(const limb_t* p, uint32_t n) {
        bigint r;
        if (n > 0) {
            r.ensure_capacity(n);
            std::memcpy(r.data_, p, n * sizeof(limb_t));
            r.set_size_sign(n, false);
        }
        return r;
    }

    // ---- Power-of-2 fast path: O(n) bit extraction ----

    static std::string pow2_to_string(const bigint& x, unsigned shift) {
        uint32_t bits = x.bit_length();
        uint32_t n_digits = (bits + shift - 1) / shift;
        std::string result;
        result.reserve(n_digits + 1);
        if (x.is_negative()) result.push_back('-');

        uint32_t mask = (1u << shift) - 1;
        const limb_t* p = x.limbs();
        uint32_t n = x.abs_size();

        for (int i = (int)n_digits - 1; i >= 0; i--) {
            uint32_t bit_pos = (uint32_t)i * shift;
            uint32_t limb_idx = bit_pos / 64;
            unsigned bit_off = bit_pos % 64;
            uint32_t val = 0;
            if (limb_idx < n) {
                val = (uint32_t)((p[limb_idx] >> bit_off) & mask);
                if (bit_off + shift > 64 && limb_idx + 1 < n) {
                    val |= (uint32_t)((p[limb_idx + 1] << (64 - bit_off)) & mask);
                }
            }
            result.push_back(RADIX_ALPHABET[val]);
        }
        return result;
    }

    static bigint pow2_from_string(const char* s, uint32_t len, unsigned shift, bool neg) {
        if (len == 0) return bigint();
        uint32_t total_bits = len * shift;
        uint32_t n_limbs = (total_bits + 63) / 64;

        bigint result;
        result.ensure_capacity(n_limbs);
        limb_t* rp = result.data_;
        mpn_zero(rp, n_limbs);

        uint32_t base = 1u << shift;
        for (uint32_t i = 0; i < len; i++) {
            int v = digit_value(s[len - 1 - i], base);
            if (v < 0) break;
            uint32_t bit_pos = i * shift;
            uint32_t limb_idx = bit_pos / 64;
            unsigned bit_off = bit_pos % 64;
            rp[limb_idx] |= (limb_t)(uint32_t)v << bit_off;
            if (bit_off + shift > 64 && limb_idx + 1 < n_limbs) {
                rp[limb_idx + 1] |= (limb_t)(uint32_t)v >> (64 - bit_off);
            }
        }
        n_limbs = mpn_normalize(rp, n_limbs);
        result.set_size_sign(n_limbs, neg && n_limbs > 0);
        return result;
    }

    // ---- Generalized basecase to_radix: O(n^2) via div_limb(chunk_pow) + LUT ----

    // Convert a single u64 chunk value to exactly chunk_k digits using LUT.
    static void limb_to_radix_string(char* buf, limb_t val,
                                      const radix_powers_cache& cache) {
        int pos = (int)cache.chunk_k;
        // LUT-accelerated: extract lut_t_ digits at a time
        while (pos >= (int)cache.lut_t_) {
            limb_t idx = val % cache.lut_pow;
            val /= cache.lut_pow;
            pos -= (int)cache.lut_t_;
            std::memcpy(buf + pos, cache.to_str_lut + idx * cache.lut_t_, cache.lut_t_);
        }
        // Handle remaining digits (when chunk_k % lut_t_ != 0)
        while (pos > 0) {
            buf[--pos] = RADIX_ALPHABET[val % cache.radix];
            val /= cache.radix;
        }
    }

    static void basecase_to_radix(std::string& out, bigint& x, uint32_t pad,
                                   const radix_powers_cache& cache) {
        // Extract chunks (each is a u64 in [0, chunk_pow))
        // Use a small stack buffer to avoid vector allocation
        limb_t chunk_buf[64]; // enough for 64 limbs * 19 digits = 1216 digits
        uint32_t n_chunks = 0;
        while (!x.is_zero()) {
            assert(n_chunks < 64);
            chunk_buf[n_chunks++] = x.div_limb(cache.chunk_pow);
        }
        if (n_chunks == 0) {
            if (pad > 0) out.append(pad, '0');
            return;
        }

        // Convert highest chunk (no leading-zero padding)
        std::string s;
        s.reserve(n_chunks * cache.chunk_k);
        {
            limb_t val = chunk_buf[n_chunks - 1];
            if (val == 0) {
                s.push_back('0');
            } else {
                char tmp[20]; // max 19 digits for base 10, chunk_k <= 64
                int pos = 0;
                while (val > 0) {
                    tmp[pos++] = RADIX_ALPHABET[val % cache.radix];
                    val /= cache.radix;
                }
                for (int i = pos - 1; i >= 0; i--)
                    s.push_back(tmp[i]);
            }
        }

        // Remaining chunks: zero-padded to chunk_k digits, using LUT
        char cbuf[64]; // chunk_k <= 64
        for (int i = (int)n_chunks - 2; i >= 0; i--) {
            limb_to_radix_string(cbuf, chunk_buf[i], cache);
            s.append(cbuf, cache.chunk_k);
        }

        if (pad > 0 && s.size() < (size_t)pad)
            out.append((size_t)pad - s.size(), '0');
        out += s;
    }

    // ---- Generalized D&C to_radix: O(M(n) log n) ----

    static void dc_to_radix(std::string& out, bigint& x,
                              uint32_t est_digits, uint32_t pad,
                              radix_powers_cache& cache) {
        if (x.is_zero()) {
            if (pad > 0) out.append(pad, '0');
            return;
        }
        if (x.abs_size() <= RADIX_DC_THRESHOLD) {
            basecase_to_radix(out, x, pad, cache);
            return;
        }

        // Tighten estimate from actual bit length
        uint32_t actual_est = est_radix_digits(x.bit_length(), cache.radix);
        if (actual_est < est_digits) est_digits = actual_est;

        // Split in chunk units: find k such that chunk_k * 2^k ~ est_digits/2
        uint32_t est_chunks = (est_digits + cache.chunk_k - 1) / cache.chunk_k;
        uint32_t half_chunks = est_chunks / 2;
        if (half_chunks < 1) half_chunks = 1;
        uint32_t k = 0;
        while ((1u << (k + 1)) <= half_chunks) k++;
        uint32_t split_digits = cache.chunk_k * (1u << k);

        const auto& pow = cache.get_pow2k(k);
        uint32_t nn = x.abs_size(), dn = pow.size;

        bigint q, r;
        if (nn < dn || (nn == dn && mpn_cmp(x.data_, pow.data, dn) < 0)) {
            r = std::move(x);
        } else {
            uint32_t qn = nn - dn + 1;
            q.ensure_capacity(qn);
            r.ensure_capacity(dn);

            cache.ensure_reciprocal(k);
            if (pow.inv) {
                mpn_tdiv_qr_preinv(q.data_, r.data_, x.data_, nn,
                                    pow.d_norm, dn, pow.inv, pow.shift);
            } else {
                mpn_tdiv_qr(q.data_, r.data_, x.data_, nn, pow.data, dn);
            }

            qn = mpn_normalize(q.data_, qn);
            uint32_t rn = mpn_normalize(r.data_, dn);
            q.set_size_sign(qn, false);
            r.set_size_sign(rn, false);
        }

        uint32_t high_pad = (pad > split_digits) ? pad - split_digits : 0;
        dc_to_radix(out, q, est_digits - split_digits, high_pad, cache);
        dc_to_radix(out, r, split_digits, split_digits, cache);
    }

    // ---- Generalized basecase from_radix: O(n^2) via chunk_k-char groups ----

    static bigint basecase_from_radix(const char* s, uint32_t len,
                                       const radix_powers_cache& cache) {
        bigint result;
        uint32_t pos = 0;
        uint32_t base = cache.radix;

        // First partial chunk (align remaining to chunk_k boundaries)
        uint32_t first = len % cache.chunk_k;
        if (first == 0 && len > 0) first = cache.chunk_k;

        limb_t val = 0;
        for (uint32_t i = 0; i < first; i++) {
            int d = digit_value(s[i], base);
            if (d < 0) d = 0;
            val = val * base + (limb_t)d;
        }
        pos = first;

        if (val > 0) {
            result.ensure_capacity(1);
            result.data_[0] = val;
            result.set_size_sign(1, false);
        }

        while (pos < len) {
            limb_t chunk = 0;
            for (uint32_t i = 0; i < cache.chunk_k; i++) {
                int d = digit_value(s[pos + i], base);
                if (d < 0) d = 0;
                chunk = chunk * base + (limb_t)d;
            }
            pos += cache.chunk_k;
            result.mul_limb(cache.chunk_pow);
            result.add_limb(chunk);
        }
        return result;
    }

    // ---- Generalized D&C from_radix: O(M(n) log n) ----

    static bigint dc_from_radix(const char* s, uint32_t len,
                                  radix_powers_cache& cache) {
        if (len == 0) return bigint();
        if (len <= cache.chunk_k * RADIX_DC_THRESHOLD)
            return basecase_from_radix(s, len, cache);

        // Split in chunk units
        uint32_t est_chunks = (len + cache.chunk_k - 1) / cache.chunk_k;
        uint32_t half_chunks = est_chunks / 2;
        if (half_chunks < 1) half_chunks = 1;
        uint32_t k = 0;
        while ((1u << (k + 1)) <= half_chunks) k++;
        uint32_t split_digits = cache.chunk_k * (1u << k);

        // Safety: ensure split doesn't exceed len
        if (split_digits >= len) {
            return basecase_from_radix(s, len, cache);
        }

        bigint high = dc_from_radix(s, len - split_digits, cache);
        bigint low  = dc_from_radix(s + (len - split_digits), split_digits, cache);

        const auto& pow = cache.get_pow2k(k);
        bigint pow_bi = from_limbs_unsigned(pow.data, pow.size);
        high *= pow_bi;
        high += low;
        return high;
    }

public:
    // ---- Multiply placeholder (full implementation in mul.hpp) ----

    bigint& operator*=(long long val) {
        if (is_zero() || val == 0) {
            size_ = 0;
            return *this;
        }
        bool flip_sign = (val < 0);
        uint64_t abs_val = flip_sign ? (uint64_t)(-(val + 1)) + 1u : (uint64_t)val;
        mul_limb(abs_val);
        if (flip_sign) negate();
        return *this;
    }

    bigint operator*(long long val) const { bigint r(*this); r *= val; return r; }

    // ---- Full multiplication (bigint * bigint) ----

    bigint& operator*=(const bigint& o) {
        if (is_zero() || o.is_zero()) {
            size_ = 0;
            return *this;
        }
        bool neg = is_negative() != o.is_negative();
        uint32_t an = abs_size(), bn = o.abs_size();

        uint32_t rn = an + bn;
        limb_t* rp = mpn_alloc(rn);

        if (this == &o) {
            mpn_sqr(rp, data_, an);
        } else if (an >= bn) {
            mpn_mul(rp, data_, an, o.data_, bn);
        } else {
            mpn_mul(rp, o.data_, bn, data_, an);
        }

        mpn_free(data_);
        data_ = rp;
        alloc_ = rn;
        rn = mpn_normalize(rp, rn);
        set_size_sign(rn, neg);
        return *this;
    }

    bigint operator*(const bigint& o) const { bigint r(*this); r *= o; return r; }

    // ---- Division and modulo ----

    // Truncated division: q = trunc(a / b), r = a - q*b (sign of r = sign of a)
    static void divmod(const bigint& a, const bigint& b, bigint& q, bigint& r) {
        assert(!b.is_zero());
        if (a.is_zero()) { q = bigint(); r = bigint(); return; }

        int cmp = a.compare_abs(b);
        if (cmp < 0) {
            // |a| < |b|: quotient = 0, remainder = a
            q = bigint();
            r = a;
            return;
        }
        if (cmp == 0) {
            // |a| == |b|: quotient = ±1, remainder = 0
            q = bigint(a.is_negative() == b.is_negative() ? 1LL : -1LL);
            r = bigint();
            return;
        }

        bool q_neg = a.is_negative() != b.is_negative();
        bool r_neg = a.is_negative();

        uint32_t nn = a.abs_size(), dn = b.abs_size();
        uint32_t qn = nn - dn + 1;

        q.ensure_capacity(qn);
        r.ensure_capacity(dn);

        mpn_tdiv_qr(q.data_, r.data_, a.data_, nn, b.data_, dn);

        qn = mpn_normalize(q.data_, qn);
        uint32_t rn = mpn_normalize(r.data_, dn);

        q.set_size_sign(qn, q_neg && qn > 0);
        r.set_size_sign(rn, r_neg && rn > 0);
    }

    bigint& operator/=(const bigint& o) {
        bigint q, r;
        divmod(*this, o, q, r);
        *this = std::move(q);
        return *this;
    }

    bigint& operator%=(const bigint& o) {
        bigint q, r;
        divmod(*this, o, q, r);
        *this = std::move(r);
        return *this;
    }

    bigint operator/(const bigint& o) const { bigint r(*this); r /= o; return r; }
    bigint operator%(const bigint& o) const { bigint q, r; divmod(*this, o, q, r); return r; }

    // Single-limb division
    bigint& operator/=(long long val) { bigint tmp(val); *this /= tmp; return *this; }
    bigint& operator%=(long long val) { bigint tmp(val); *this %= tmp; return *this; }
    bigint operator/(long long val) const { bigint r(*this); r /= val; return r; }
    bigint operator%(long long val) const { bigint r(*this); r %= val; return r; }
};

} // namespace zint
