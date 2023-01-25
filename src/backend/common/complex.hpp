/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <backend.hpp>
#include <types.hpp>

#include <type_traits>

namespace arrayfire {
namespace common {

// The value returns true if the type is a complex type. False otherwise
template<typename T>
struct is_complex {
    static const bool value = false;
};
template<>
struct is_complex<detail::cfloat> {
    static const bool value = true;
};
template<>
struct is_complex<detail::cdouble> {
    static const bool value = true;
};

/// This is an enable_if for complex types.
template<typename T, typename TYPE = void>
using if_complex = typename std::enable_if<is_complex<T>::value, TYPE>::type;

/// This is an enable_if for real types.
template<typename T, typename TYPE = void>
using if_real =
    typename std::enable_if<is_complex<T>::value == false, TYPE>::type;

}  // namespace common
}  // namespace arrayfire
