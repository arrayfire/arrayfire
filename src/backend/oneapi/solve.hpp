/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace arrayfire {
namespace oneapi {

template<typename T, typename U, typename... Args>
static inline constexpr bool is_any_of() {
    if constexpr (!sizeof...(Args)) {
        return std::is_same_v<T, U>;
    } else {
        return std::is_same_v<T, U> || is_any_of<T, Args...>();
    }
}

template<typename T>
Array<T> solve(const Array<T> &a, const Array<T> &b,
               const af_mat_prop options = AF_MAT_NONE);

template<typename T>
Array<T> solveLU(const Array<T> &a, const Array<int> &pivot, const Array<T> &b,
                 const af_mat_prop options = AF_MAT_NONE);
}  // namespace oneapi
}  // namespace arrayfire
