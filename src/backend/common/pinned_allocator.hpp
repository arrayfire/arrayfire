/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <memory.hpp>

namespace common {
template<typename T>
class pinned_allocator {
   public:
    using value_type = T;

    pinned_allocator() noexcept {}  // not required, unless used
    template<typename U>
    pinned_allocator(std::allocator<U> const&) noexcept {}

    value_type* allocate(std::size_t n) {
        return detail::pinnedAlloc<value_type>(n);
    }

    void deallocate(value_type* p, std::size_t) noexcept {
        detail::pinnedFree(p);
    }
};

template<typename T, typename U>
bool operator==(pinned_allocator<T> const&,
                pinned_allocator<U> const&) noexcept {
    return true;
}

template<typename T, typename U>
bool operator!=(pinned_allocator<T> const& x,
                pinned_allocator<U> const& y) noexcept {
    return !(x == y);
}
}  // namespace common
