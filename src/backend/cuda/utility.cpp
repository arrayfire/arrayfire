/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <err_cuda.hpp>
#include <utility.hpp>

using std::make_pair;
using std::pair;

namespace cuda {

pair<InterpolationType, int> toInternalEnum(const af_interp_type p) {
    InterpolationType method = InterpolationType::Nearest;
    int order                = 1;
    switch (p) {
        case AF_INTERP_NEAREST:
            method = InterpolationType::Nearest;
            order  = 1;
            break;
        case AF_INTERP_LOWER:
            method = InterpolationType::Lower;
            order  = 1;
            break;
        case AF_INTERP_LINEAR:
            method = InterpolationType::Linear;
            order  = 2;
            break;
        case AF_INTERP_BILINEAR:
            method = InterpolationType::Bilinear;
            order  = 2;
            break;
        case AF_INTERP_LINEAR_COSINE:
            method = InterpolationType::LinearCosine;
            order  = 2;
            break;
        case AF_INTERP_BILINEAR_COSINE:
            method = InterpolationType::BilinearCosine;
            order  = 2;
            break;
        case AF_INTERP_CUBIC:
            method = InterpolationType::Cubic;
            order  = 3;
            break;
        case AF_INTERP_BICUBIC:
            method = InterpolationType::Bicubic;
            order  = 3;
            break;
        case AF_INTERP_CUBIC_SPLINE:
            method = InterpolationType::CubicSpline;
            order  = 3;
            break;
        case AF_INTERP_BICUBIC_SPLINE:
            method = InterpolationType::BicubicSpline;
            order  = 3;
            break;
        default: AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
    }
    return make_pair(method, order);
}

BorderType toInternalEnum(const af_border_type p) {
    auto retVal = BorderType::Zero;
    switch (p) {
        case AF_PAD_SYM: retVal = BorderType::Symmetric; break;
        case AF_PAD_CLAMP_TO_EDGE: retVal = BorderType::ClampToEdge; break;
        case AF_PAD_ZERO:
        default: retVal = BorderType::Zero; break;
    }
    return retVal;
}

MomentType toInternalEnum(const af_moment_type p) {
    auto retVal = MomentType::FirstOrder;
    switch (p) {
        case AF_MOMENT_M00: retVal = MomentType::M00; break;
        case AF_MOMENT_M01: retVal = MomentType::M01; break;
        case AF_MOMENT_M10: retVal = MomentType::M10; break;
        case AF_MOMENT_M11: retVal = MomentType::M11; break;
        case AF_MOMENT_FIRST_ORDER:
        default: retVal = MomentType::FirstOrder;
    }
    return retVal;
}

FluxFunction toInternalEnum(const af_flux_function p) {
    auto retVal = FluxFunction::Default;
    switch (p) {
        case AF_FLUX_QUADRATIC: retVal = FluxFunction::Quadratic; break;
        case AF_FLUX_EXPONENTIAL: retVal = FluxFunction::Exponential; break;
        default: retVal = FluxFunction::Default;
    }
    return retVal;
}

ErrorMetric toInternalEnum(const af_match_type p) {
    ErrorMetric retVal;
    switch (p) {
        case AF_ZSAD: retVal = ErrorMetric::ZSAD; break;
        case AF_LSAD: retVal = ErrorMetric::LSAD; break;
        case AF_SSD: retVal = ErrorMetric::SSD; break;
        case AF_ZSSD: retVal = ErrorMetric::ZSSD; break;
        case AF_LSSD: retVal = ErrorMetric::LSSD; break;
        case AF_NCC: retVal = ErrorMetric::NCC; break;
        case AF_ZNCC: retVal = ErrorMetric::ZNCC; break;
        case AF_SAD:
        default: retVal = ErrorMetric::SAD; break;
    }
    return retVal;
}

}  // namespace cuda
