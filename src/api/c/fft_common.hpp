/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <copy.hpp>
#include <fft.hpp>
#include <handle.hpp>

void computePaddedDims(af::dim4 &pdims, const af::dim4 &idims, const dim_t npad,
                       dim_t const *const pad);

template<typename inType, typename outType>
detail::Array<outType> fft(const detail::Array<inType> input,
                           const double norm_factor, const dim_t npad,
                           const dim_t *const pad, const int rank,
                           const bool direction) {
    using af::dim4;
    using detail::fft_inplace;
    using detail::reshape;
    using detail::scalar;

    dim4 pdims(1);
    computePaddedDims(pdims, input.dims(), npad, pad);
    auto res = reshape(input, pdims, scalar<outType>(0));

    fft_inplace<outType>(res, rank, direction);
    if (norm_factor != 1.0) multiply_inplace(res, norm_factor);

    return res;
}

template<typename inType, typename outType>
detail::Array<outType> fft_r2c(const detail::Array<inType> input,
                               const double norm_factor, const dim_t npad,
                               const dim_t *const pad, const int rank) {
    using af::dim4;
    using detail::Array;
    using detail::fft_r2c;
    using detail::multiply_inplace;
    using detail::reshape;
    using detail::scalar;

    const dim4 &idims = input.dims();

    bool is_pad = false;
    for (int i = 0; i < npad; i++) { is_pad |= (pad[i] != idims[i]); }

    Array<inType> tmp = input;

    if (is_pad) {
        dim4 pdims(1);
        computePaddedDims(pdims, input.dims(), npad, pad);
        tmp = reshape(input, pdims, scalar<inType>(0));
    }

    auto res = fft_r2c<outType, inType>(tmp, rank);
    if (norm_factor != 1.0) multiply_inplace(res, norm_factor);

    return res;
}

template<typename inType, typename outType>
detail::Array<outType> fft_c2r(const detail::Array<inType> input,
                               const double norm_factor, const af::dim4 &odims,
                               const int rank) {
    using detail::Array;
    using detail::fft_c2r;
    using detail::multiply_inplace;

    Array<outType> output = fft_c2r<outType, inType>(input, odims, rank);

    if (norm_factor != 1) {
        // Normalize input because tmp was not normalized
        multiply_inplace(output, norm_factor);
    }

    return output;
}
