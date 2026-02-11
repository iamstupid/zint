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
        *this = bitwise_binary_op(*this, o, bitwise_op::and_);
        return *this;
    }

    bigint& operator|=(const bigint& o) {
        *this = bitwise_binary_op(*this, o, bitwise_op::or_);
        return *this;
    }

    bigint& operator^=(const bigint& o) {
        *this = bitwise_binary_op(*this, o, bitwise_op::xor_);
        return *this;
    }

    bigint operator&(const bigint& o) const { bigint r(*this); r &= o; return r; }
    bigint operator|(const bigint& o) const { bigint r(*this); r |= o; return r; }
    bigint operator^(const bigint& o) const { bigint r(*this); r ^= o; return r; }

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

    // ---- Radix power cache (optional) ----
    //
    // Stores powers base^(2^k) lazily. Used by radix conversion routines
    // (notably base-10 D&C conversion). By default, to_string/from_string will
    // compute and free powers on-the-fly; users may inject a cache to reuse
    // powers across repeated conversions.
    //
    // Not thread-safe if shared across threads.
    struct radix_powers_cache {
        struct entry {
            limb_t* data = nullptr;
            uint32_t size = 0;
        };

        limb_t base = 0;
        std::vector<entry> pow2k{}; // pow2k[k] = base^(2^k)

        radix_powers_cache() = default;
        explicit radix_powers_cache(limb_t b) { reset(b); }

        radix_powers_cache(const radix_powers_cache&) = delete;
        radix_powers_cache& operator=(const radix_powers_cache&) = delete;

        radix_powers_cache(radix_powers_cache&& o) noexcept
            : base(o.base), pow2k(std::move(o.pow2k)) {
            o.base = 0;
            o.pow2k.clear();
        }
        radix_powers_cache& operator=(radix_powers_cache&& o) noexcept {
            if (this != &o) {
                clear();
                base = o.base;
                pow2k = std::move(o.pow2k);
                o.base = 0;
                o.pow2k.clear();
            }
            return *this;
        }

        ~radix_powers_cache() { clear(); }

        void clear() noexcept {
            for (auto& e : pow2k) {
                mpn_free(e.data);
                e.data = nullptr;
                e.size = 0;
            }
            pow2k.clear();
            base = 0;
        }

        void reset(limb_t b) {
            if (base == b && !pow2k.empty()) return;
            clear();
            base = b;
            if (base == 0) return;
            limb_t* d = mpn_alloc(1);
            d[0] = base;
            pow2k.push_back({d, 1});
        }

        const entry& get_pow2k(uint32_t k) {
            assert(base != 0);
            while (k >= (uint32_t)pow2k.size()) {
                const entry& prev = pow2k.back();
                uint32_t rn = 2 * prev.size;
                limb_t* rp = mpn_alloc(rn);
                mpn_sqr(rp, prev.data, prev.size);
                rn = mpn_normalize(rp, rn);
                pow2k.push_back({rp, rn});
            }
            return pow2k[k];
        }
    };

    // ---- String conversion (O(n^2) basecase) ----

    std::string to_string(int base = 10, radix_powers_cache* pow_cache = nullptr) const {
        if (is_zero()) return "0";
        assert(base >= 2 && base <= 36);

        // For base 10, use chunks of 10^18
        if (base == 10) return to_decimal_string(pow_cache);

        // Generic base conversion
        bigint tmp = this->abs();
        std::string result;
        static const char digits[] = "0123456789abcdefghijklmnopqrstuvwxyz";

        while (!tmp.is_zero()) {
            limb_t rem = tmp.div_limb((limb_t)base);
            result.push_back(digits[rem]);
        }

        if (is_negative()) result.push_back('-');
        std::reverse(result.begin(), result.end());
        return result;
    }

    // Construct from decimal string
    static bigint from_string(const char* s, int base = 10, radix_powers_cache* pow_cache = nullptr) {
        assert(base >= 2 && base <= 36);
        if (!s || !*s) return bigint();

        bool neg = false;
        if (*s == '-') { neg = true; s++; }
        else if (*s == '+') { s++; }

        if (base == 10) return from_decimal_string(s, neg, pow_cache);

        bigint result;
        for (; *s; s++) {
            int d;
            char c = *s;
            if (c >= '0' && c <= '9') d = c - '0';
            else if (c >= 'a' && c <= 'z') d = c - 'a' + 10;
            else if (c >= 'A' && c <= 'Z') d = c - 'A' + 10;
            else break; // stop at non-digit
            if (d >= base) break;
            result.mul_limb((limb_t)base);
            result.add_limb((limb_t)d);
        }
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

    static constexpr limb_t POW10_18 = 1000000000000000000ULL; // 10^18
    static constexpr int    DIG10_18 = 18;
    static constexpr uint32_t RADIX_DC_THRESHOLD = 30; // limbs: D&C above this

    // Powers of 10 are provided via radix_powers_cache (on-the-fly by default).

    // Upper bound on decimal digits for a number with 'bits' bits
    static uint32_t est_decimal_digits(uint32_t bits) {
        if (bits == 0) return 1;
        return (uint32_t)(((uint64_t)bits * 78 + 255) / 256) + 1;
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

    // ---- Basecase to_decimal: O(n^2) via repeated div by 10^18 ----
    static void basecase_to_decimal(std::string& out, bigint& x, uint32_t pad) {
        std::vector<std::string> groups;
        while (!x.is_zero()) {
            limb_t rem = x.div_limb(POW10_18);
            groups.push_back(std::to_string(rem));
        }
        if (groups.empty()) {
            if (pad > 0) out.append(pad, '0');
            return;
        }
        std::string s = groups.back();
        for (int i = (int)groups.size() - 2; i >= 0; i--) {
            const std::string& g = groups[i];
            if (g.size() < (size_t)DIG10_18)
                s.append((size_t)DIG10_18 - g.size(), '0');
            s += g;
        }
        if (pad > 0 && s.size() < (size_t)pad)
            out.append((size_t)pad - s.size(), '0');
        out += s;
    }

    // ---- D&C to_decimal: O(M(n) log n) ----
    static void dc_to_decimal(std::string& out, bigint& x,
                               uint32_t est_digits, uint32_t pad,
                               radix_powers_cache& pow_cache) {
        if (x.is_zero()) {
            if (pad > 0) out.append(pad, '0');
            return;
        }
        if (x.abs_size() <= RADIX_DC_THRESHOLD) {
            basecase_to_decimal(out, x, pad);
            return;
        }

        // Tighten estimate from actual bit length
        uint32_t actual_est = est_decimal_digits(x.bit_length());
        if (actual_est < est_digits) est_digits = actual_est;

        uint32_t half = est_digits / 2;
        if (half < 1) half = 1;
        uint32_t k = 0;
        while ((1u << (k + 1)) <= half) k++;
        uint32_t split = 1u << k;

        const auto& pow = pow_cache.get_pow2k(k);
        bigint pow_bi = from_limbs_unsigned(pow.data, pow.size);
        bigint q, r;
        divmod(x, pow_bi, q, r);

        // High part (quotient)
        uint32_t high_pad = (pad > split) ? pad - split : 0;
        dc_to_decimal(out, q, est_digits - split, high_pad, pow_cache);
        // Low part (remainder, zero-padded to split digits)
        dc_to_decimal(out, r, split, split, pow_cache);
    }

    std::string to_decimal_string(radix_powers_cache* pow_cache) const {
        uint32_t nn = abs_size();
        if (nn == 0) return "0";

        bigint tmp = this->abs();
        std::string result;
        if (is_negative()) result.push_back('-');

        uint32_t est = est_decimal_digits(bit_length());
        radix_powers_cache local;
        if (!pow_cache) {
            local.reset(10);
            pow_cache = &local;
        } else {
            if (pow_cache->base == 0) pow_cache->reset(10);
            else assert(pow_cache->base == 10);
        }
        dc_to_decimal(result, tmp, est, 0, *pow_cache);
        return result;
    }

    // ---- Basecase from_decimal: O(n^2) via 18-digit chunks ----
    static bigint basecase_from_decimal(const char* s, uint32_t len) {
        bigint result;
        uint32_t pos = 0;

        // First partial chunk (align remaining to 18-digit boundaries)
        uint32_t first = len % DIG10_18;
        if (first == 0 && len > 0) first = DIG10_18;

        limb_t val = 0;
        for (uint32_t i = 0; i < first; i++)
            val = val * 10 + (s[i] - '0');
        pos = first;

        if (val > 0) {
            result.ensure_capacity(1);
            result.data_[0] = val;
            result.set_size_sign(1, false);
        }

        while (pos < len) {
            limb_t chunk = 0;
            for (int i = 0; i < DIG10_18; i++)
                chunk = chunk * 10 + (s[pos + i] - '0');
            pos += DIG10_18;
            result.mul_limb(POW10_18);
            result.add_limb(chunk);
        }
        return result;
    }

    // ---- D&C from_decimal: O(M(n) log n) ----
    static bigint dc_from_decimal(const char* s, uint32_t len, radix_powers_cache& pow_cache) {
        if (len == 0) return bigint();
        if (len <= (uint32_t)DIG10_18 * RADIX_DC_THRESHOLD)
            return basecase_from_decimal(s, len);

        // Split: low part has 2^k digits, high part has (len - 2^k) digits
        uint32_t half = len / 2;
        uint32_t k = 0;
        while ((1u << (k + 1)) <= half) k++;
        uint32_t split = 1u << k;

        bigint high = dc_from_decimal(s, len - split, pow_cache);
        bigint low  = dc_from_decimal(s + (len - split), split, pow_cache);

        const auto& pow = pow_cache.get_pow2k(k);
        bigint pow_bi = from_limbs_unsigned(pow.data, pow.size);
        high *= pow_bi;
        high += low;
        return high;
    }

    static bigint from_decimal_string(const char* s, bool neg, radix_powers_cache* pow_cache) {
        size_t len = std::strlen(s);
        if (len == 0) return bigint();

        // Skip leading zeros
        while (len > 1 && *s == '0') { s++; len--; }

        radix_powers_cache local;
        if (!pow_cache) {
            local.reset(10);
            pow_cache = &local;
        } else {
            if (pow_cache->base == 0) pow_cache->reset(10);
            else assert(pow_cache->base == 10);
        }

        bigint result = dc_from_decimal(s, (uint32_t)len, *pow_cache);
        if (neg && !result.is_zero()) result.negate();
        return result;
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
