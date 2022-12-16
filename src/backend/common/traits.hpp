/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <common/err_common.hpp>
#include <af/defines.h>

namespace af {
template<typename T>
struct dtype_traits;
}

namespace arrayfire {
namespace common {
class half;

namespace {

inline size_t dtypeSize(af::dtype type) {
    switch (type) {
        case u8:
        case b8: return 1;
        case s16:
        case u16:
        case f16: return 2;
        case s32:
        case u32:
        case f32: return 4;
        case u64:
        case s64:
        case c32:
        case f64: return 8;
        case c64: return 16;
        default: AF_RETURN_ERROR("Unsupported type", AF_ERR_INTERNAL);
    }
}

constexpr bool isComplex(af::dtype type) {
    return ((type == c32) || (type == c64));
}

constexpr bool isReal(af::dtype type) { return !isComplex(type); }

constexpr bool isDouble(af::dtype type) { return (type == f64 || type == c64); }

constexpr bool isSingle(af::dtype type) { return (type == f32 || type == c32); }

constexpr bool isHalf(af::dtype type) { return (type == f16); }

constexpr bool isRealFloating(af::dtype type) {
    return (type == f64 || type == f32 || type == f16);
}

constexpr bool isInteger(af::dtype type) {
    return (type == s32 || type == u32 || type == s64 || type == u64 ||
            type == s16 || type == u16 || type == u8);
}

constexpr bool isBool(af::dtype type) { return (type == b8); }

constexpr bool isFloating(af::dtype type) {
    return (!isInteger(type) && !isBool(type));
}

}  // namespace
}  // namespace common
}  // namespace arrayfire

namespace af {
template<>
struct dtype_traits<arrayfire::common::half> {
    enum { af_type = f16, ctype = f16 };
    typedef arrayfire::common::half base_type;
    static const char *getName() { return "half"; }
};
}  // namespace af
