/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <anisotropic_diffusion.hpp>

#include <arith.hpp>
#include <backend.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <gradient.hpp>
#include <handle.hpp>
#include <reduce.hpp>

#include <af/dim4.hpp>
#include <af/image.h>

#include <type_traits>

using af::dim4;
using arrayfire::common::cast;
using detail::arithOp;
using detail::Array;
using detail::createEmptyArray;
using detail::gradient;
using detail::reduce_all;

template<typename T>
af_array diffusion(const Array<float>& in, const float dt, const float K,
                   const unsigned iterations, const af_flux_function fftype,
                   const af::diffusionEq eq) {
    auto out  = copyArray(in);
    auto dims = out.dims();
    auto g0   = createEmptyArray<float>(dims);
    auto g1   = createEmptyArray<float>(dims);
    float cnst =
        -2.0f * K * K / dims.elements();  // NOLINT(readability-magic-numbers)

    for (unsigned i = 0; i < iterations; ++i) {
        gradient<float>(g0, g1, out);

        auto g0Sqr = arithOp<float, af_mul_t>(g0, g0, dims);
        auto g1Sqr = arithOp<float, af_mul_t>(g1, g1, dims);
        auto sumd  = arithOp<float, af_add_t>(g0Sqr, g1Sqr, dims);
        float avg  = reduce_all<af_add_t, float, float>(sumd, true, 0);

        anisotropicDiffusion(out, dt, 1.0f / (cnst * avg), fftype, eq);
    }

    return getHandle(cast<T, float>(out));
}

af_err af_anisotropic_diffusion(af_array* out, const af_array in,
                                const float dt, const float K,
                                const unsigned iterations,
                                const af_flux_function fftype,
                                const af_diffusion_eq eq) {
    try {
        const ArrayInfo& info = getInfo(in);

        const af::dim4& inputDimensions = info.dims();
        const af_dtype inputType        = info.getType();
        const unsigned inputNumDims     = inputDimensions.ndims();

        DIM_ASSERT(1, (inputNumDims >= 2));

        ARG_ASSERT(3, (K > 0 || K < 0));
        ARG_ASSERT(4, (iterations > 0));

        const af_flux_function F =
            (fftype == AF_FLUX_DEFAULT ? AF_FLUX_EXPONENTIAL : fftype);

        auto input = castArray<float>(in);

        af_array output = nullptr;
        switch (inputType) {
            case f64:
                output = diffusion<double>(input, dt, K, iterations, F, eq);
                break;
            case f32:
            case s32:
            case u32:
            case s16:
            case u16:
            case u8:
                output = diffusion<float>(input, dt, K, iterations, F, eq);
                break;
            default: TYPE_ERROR(1, inputType);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
