#pragma once
// scratch.hpp - Thread-local scratchpad with mark/restore (bump allocator)
//
// Design goals:
// - Fast temp allocations for bigint and NTT (no per-call malloc/free in hot paths)
// - Late free: memory is released only at thread exit
// - Nested scopes via mark/restore (stack discipline)

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

namespace zint {

class Scratch {
public:
    struct Mark {
        std::size_t block_index = 0;
        std::size_t offset = 0;
    };

    Scratch(std::size_t default_block_bytes = (1u << 20), std::size_t block_align = 4096)
        : default_block_bytes_(default_block_bytes), block_align_(block_align) {}

    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;

    ~Scratch() {
        for (auto& b : blocks_) {
            if (b.ptr) ::operator delete(b.ptr, std::align_val_t(block_align_));
        }
    }

    Mark mark() const noexcept {
        if (blocks_.empty()) return {};
        return Mark{active_, blocks_[active_].off};
    }

    void restore(Mark m) noexcept {
        if (blocks_.empty()) return;
        assert(m.block_index < blocks_.size());
        for (std::size_t i = m.block_index + 1; i < blocks_.size(); ++i) {
            blocks_[i].off = 0;
        }
        blocks_[m.block_index].off = m.offset;
        active_ = m.block_index;
    }

    void* alloc_bytes(std::size_t bytes, std::size_t align) {
        if (bytes == 0) return nullptr;
        if (align < alignof(void*)) align = alignof(void*);
        assert(is_pow2(align));

        // Try current and subsequent blocks (subsequent blocks are reset on restore()).
        for (std::size_t i = active_; i < blocks_.size(); ++i) {
            Block& b = blocks_[i];
            std::size_t off = align_up(b.off, align);
            if (off + bytes <= b.cap) {
                active_ = i;
                b.off = off + bytes;
                return b.ptr + off;
            }
        }

        // Need a new block.
        grow(bytes, align);
        Block& b = blocks_.back();
        std::size_t off = align_up(b.off, align);
        assert(off + bytes <= b.cap);
        b.off = off + bytes;
        active_ = blocks_.size() - 1;
        return b.ptr + off;
    }

    template<class T>
    T* alloc(std::size_t count, std::size_t align = alignof(T)) {
        static_assert(std::is_trivially_destructible_v<T>, "Scratch only supports POD-like types");
        if (count == 0) return nullptr;
        if (count > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) {
            assert(false && "Scratch allocation overflow");
            return nullptr;
        }
        return static_cast<T*>(alloc_bytes(count * sizeof(T), (std::max)(align, alignof(T))));
    }

    void reserve_bytes(std::size_t bytes, std::size_t align = alignof(void*)) {
        if (bytes == 0) return;
        if (blocks_.empty()) {
            grow(bytes, align);
            return;
        }
        Block& b = blocks_[active_];
        std::size_t off = align_up(b.off, align);
        if (off + bytes <= b.cap) return;
        grow(bytes, align);
    }

private:
    struct Block {
        std::byte* ptr = nullptr;
        std::size_t cap = 0;
        std::size_t off = 0;
    };

    std::vector<Block> blocks_{};
    std::size_t active_ = 0;
    std::size_t default_block_bytes_ = 0;
    std::size_t block_align_ = 0;

    static bool is_pow2(std::size_t x) noexcept {
        return x != 0 && (x & (x - 1)) == 0;
    }

    static std::size_t align_up(std::size_t x, std::size_t align) noexcept {
        assert(is_pow2(align));
        return (x + (align - 1)) & ~(align - 1);
    }

    void grow(std::size_t min_bytes, std::size_t align) {
        (void)align;
        std::size_t want = min_bytes;
        if (want < default_block_bytes_) want = default_block_bytes_;

        if (!blocks_.empty()) {
            std::size_t prev = blocks_.back().cap;
            if (want < prev) want = prev;
            // Geometric growth to keep number of OS allocations low.
            want = (std::max)(want, prev + (prev >> 1)); // *1.5
        }

        want = align_up(want, block_align_);
        if (want == 0) want = block_align_;

        Block b;
        b.ptr = static_cast<std::byte*>(::operator new(want, std::align_val_t(block_align_)));
        b.cap = want;
        b.off = 0;
        blocks_.push_back(b);
        active_ = blocks_.size() - 1;
    }
};

class ScratchScope {
public:
    explicit ScratchScope(Scratch& s) noexcept : scratch_(&s), mark_(s.mark()) {}
    ScratchScope(const ScratchScope&) = delete;
    ScratchScope& operator=(const ScratchScope&) = delete;
    ~ScratchScope() { scratch_->restore(mark_); }

    template<class T>
    T* alloc(std::size_t count, std::size_t align = alignof(T)) {
        return scratch_->alloc<T>(count, align);
    }

private:
    Scratch* scratch_;
    Scratch::Mark mark_;
};

inline Scratch& scratch() {
    static thread_local Scratch s{};
    return s;
}

} // namespace zint

