/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifndef __CUDACC_RTC__
#include <af/defines.h>
#endif  // __CUDACC_RTC__

namespace cuda {

enum class InterpolationType {
    Nearest,
    Linear,
    Bilinear,
    Cubic,
    Lower
#if AF_API_VERSION >= 34
    ,
    LinearCosine
#endif
#if AF_API_VERSION >= 34
    ,
    BilinearCosine
#endif
#if AF_API_VERSION >= 34
    ,
    Bicubic
#endif
#if AF_API_VERSION >= 34
    ,
    CubicSpline
#endif
#if AF_API_VERSION >= 34
    ,
    BicubicSpline
#endif
};

enum class BorderType { Zero, Symmetric, ClampToEdge };

#if AF_API_VERSION >= 34
enum class MomentType : int {
    M00        = 0x0001,
    M01        = 0x0002,
    M10        = 0x0004,
    M11        = 0x0008,
    FirstOrder = M00 | M01 | M10 | M11  // 0x000F
};

inline bool operator&(MomentType x, MomentType y) {
    return (static_cast<int>(x) & static_cast<int>(y)) > 0;
}
#endif

#if AF_API_VERSION >= 36
enum class FluxFunction { Quadratic, Exponential, Default };
#endif

enum class ErrorMetric {
    SAD,   ///< Match based on Sum of Absolute Differences (SAD)
    ZSAD,  ///< Match based on Zero mean SAD
    LSAD,  ///< Match based on Locally scaled SAD
    SSD,   ///< Match based on Sum of Squared Differences (SSD)
    ZSSD,  ///< Match based on Zero mean SSD
    LSSD,  ///< Match based on Locally scaled SSD
    NCC,   ///< Match based on Normalized Cross Correlation (NCC)
    ZNCC,  ///< Match based on Zero mean NCC
    SHD    ///< Match based on Sum of Hamming Distances (SHD)
};

}  // namespace cuda
