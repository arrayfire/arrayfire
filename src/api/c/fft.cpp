/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/err_common.hpp>
#include <fft_common.hpp>
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/signal.h>

#include <type_traits>

using af::dim4;
using detail::Array;
using detail::cdouble;
using detail::cfloat;
using detail::multiply_inplace;
using std::conditional;
using std::is_same;

void computePaddedDims(dim4 &pdims, const dim4 &idims, const dim_t npad,
                       dim_t const *const pad) {
    for (int i = 0; i < 4; i++) {
        pdims[i] = (i < static_cast<int>(npad)) ? pad[i] : idims[i];
    }
}

template<typename InType>
af_array fft(const af_array in, const double norm_factor, const dim_t npad,
             const dim_t *const pad, int rank, bool direction) {
    using OutType = typename conditional<is_same<InType, double>::value ||
                                             is_same<InType, cdouble>::value,
                                         cdouble, cfloat>::type;
    return getHandle(fft<InType, OutType>(getArray<InType>(in), norm_factor,
                                          npad, pad, rank, direction));
}

af_err fft(af_array *out, const af_array in, const double norm_factor,
           const dim_t npad, const dim_t *const pad, const int rank,
           const bool direction) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();
        const dim4 &dims      = info.dims();

        if (dims.ndims() == 0) { return af_retain_array(out, in); }

        DIM_ASSERT(1, (dims.ndims() >= rank));

        af_array output;
        switch (type) {
            case c32:
                output =
                    fft<cfloat>(in, norm_factor, npad, pad, rank, direction);
                break;
            case c64:
                output =
                    fft<cdouble>(in, norm_factor, npad, pad, rank, direction);
                break;
            case f32:
                output =
                    fft<float>(in, norm_factor, npad, pad, rank, direction);
                break;
            case f64:
                output =
                    fft<double>(in, norm_factor, npad, pad, rank, direction);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fft(af_array *out, const af_array in, const double norm_factor,
              const dim_t pad0) {
    const dim_t pad[1] = {pad0};
    return fft(out, in, norm_factor, (pad0 > 0 ? 1 : 0), pad, 1, true);
}

af_err af_fft2(af_array *out, const af_array in, const double norm_factor,
               const dim_t pad0, const dim_t pad1) {
    const dim_t pad[2] = {pad0, pad1};
    return fft(out, in, norm_factor, (pad0 > 0 && pad1 > 0 ? 2 : 0), pad, 2,
               true);
}

af_err af_fft3(af_array *out, const af_array in, const double norm_factor,
               const dim_t pad0, const dim_t pad1, const dim_t pad2) {
    const dim_t pad[3] = {pad0, pad1, pad2};
    return fft(out, in, norm_factor, (pad0 > 0 && pad1 > 0 && pad2 > 0 ? 3 : 0),
               pad, 3, true);
}

af_err af_ifft(af_array *out, const af_array in, const double norm_factor,
               const dim_t pad0) {
    const dim_t pad[1] = {pad0};
    return fft(out, in, norm_factor, (pad0 > 0 ? 1 : 0), pad, 1, false);
}

af_err af_ifft2(af_array *out, const af_array in, const double norm_factor,
                const dim_t pad0, const dim_t pad1) {
    const dim_t pad[2] = {pad0, pad1};
    return fft(out, in, norm_factor, (pad0 > 0 && pad1 > 0 ? 2 : 0), pad, 2,
               false);
}

af_err af_ifft3(af_array *out, const af_array in, const double norm_factor,
                const dim_t pad0, const dim_t pad1, const dim_t pad2) {
    const dim_t pad[3] = {pad0, pad1, pad2};
    return fft(out, in, norm_factor, (pad0 > 0 && pad1 > 0 && pad2 > 0 ? 3 : 0),
               pad, 3, false);
}

template<typename T>
void fft_inplace(af_array in, const double norm_factor, int rank,
                 bool direction) {
    Array<T> &input = getArray<T>(in);
    fft_inplace<T>(input, rank, direction);
    if (norm_factor != 1) { multiply_inplace<T>(input, norm_factor); }
}

af_err fft_inplace(af_array in, const double norm_factor, int rank,
                   bool direction) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();
        af::dim4 dims         = info.dims();

        if (dims.ndims() == 0) { return AF_SUCCESS; }
        DIM_ASSERT(1, (dims.ndims() >= rank));

        switch (type) {
            case c32:
                fft_inplace<cfloat>(in, norm_factor, rank, direction);
                break;
            case c64:
                fft_inplace<cdouble>(in, norm_factor, rank, direction);
                break;
            default: TYPE_ERROR(1, type);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fft_inplace(af_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 1, true);
}

af_err af_fft2_inplace(af_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 2, true);
}

af_err af_fft3_inplace(af_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 3, true);
}

af_err af_ifft_inplace(af_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 1, false);
}

af_err af_ifft2_inplace(af_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 2, false);
}

af_err af_ifft3_inplace(af_array in, const double norm_factor) {
    return fft_inplace(in, norm_factor, 3, false);
}

template<typename InType>
af_array fft_r2c(const af_array in, const double norm_factor, const dim_t npad,
                 const dim_t *const pad, const int rank) {
    using OutType = typename conditional<is_same<InType, double>::value,
                                         cdouble, cfloat>::type;
    return getHandle(fft_r2c<InType, OutType>(getArray<InType>(in), norm_factor,
                                              npad, pad, rank));
}

af_err fft_r2c(af_array *out, const af_array in, const double norm_factor,
               const dim_t npad, const dim_t *const pad, const int rank) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();
        af::dim4 dims         = info.dims();

        if (dims.ndims() == 0) { return af_retain_array(out, in); }
        DIM_ASSERT(1, (dims.ndims() >= rank));

        af_array output;
        switch (type) {
            case f32:
                output = fft_r2c<float>(in, norm_factor, npad, pad, rank);
                break;
            case f64:
                output = fft_r2c<double>(in, norm_factor, npad, pad, rank);
                break;
            default: {
                TYPE_ERROR(1, type);
            }
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fft_r2c(af_array *out, const af_array in, const double norm_factor,
                  const dim_t pad0) {
    const dim_t pad[1] = {pad0};
    return fft_r2c(out, in, norm_factor, (pad0 > 0 ? 1 : 0), pad, 1);
}

af_err af_fft2_r2c(af_array *out, const af_array in, const double norm_factor,
                   const dim_t pad0, const dim_t pad1) {
    const dim_t pad[2] = {pad0, pad1};
    return fft_r2c(out, in, norm_factor, (pad0 > 0 && pad1 > 0 ? 2 : 0), pad,
                   2);
}

af_err af_fft3_r2c(af_array *out, const af_array in, const double norm_factor,
                   const dim_t pad0, const dim_t pad1, const dim_t pad2) {
    const dim_t pad[3] = {pad0, pad1, pad2};
    return fft_r2c(out, in, norm_factor,
                   (pad0 > 0 && pad1 > 0 && pad2 > 0 ? 3 : 0), pad, 3);
}

template<typename InType>
static af_array fft_c2r(const af_array in, const double norm_factor,
                        const dim4 &odims, const int rank) {
    using OutType = typename conditional<is_same<InType, cdouble>::value,
                                         double, float>::type;
    return getHandle(fft_c2r<InType, OutType>(getArray<InType>(in), norm_factor,
                                              odims, rank));
}

af_err fft_c2r(af_array *out, const af_array in, const double norm_factor,
               const bool is_odd, const int rank) {
    try {
        const ArrayInfo &info = getInfo(in);
        af_dtype type         = info.getType();
        af::dim4 idims        = info.dims();

        if (idims.ndims() == 0) { return af_retain_array(out, in); }
        DIM_ASSERT(1, (idims.ndims() >= rank));

        dim4 odims = idims;
        odims[0]   = 2 * (odims[0] - 1) + (is_odd ? 1 : 0);

        af_array output;
        switch (type) {
            case c32:
                output = fft_c2r<cfloat>(in, norm_factor, odims, rank);
                break;
            case c64:
                output = fft_c2r<cdouble>(in, norm_factor, odims, rank);
                break;
            default: TYPE_ERROR(1, type);
        }
        std::swap(*out, output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_fft_c2r(af_array *out, const af_array in, const double norm_factor,
                  const bool is_odd) {
    return fft_c2r(out, in, norm_factor, is_odd, 1);
}

af_err af_fft2_c2r(af_array *out, const af_array in, const double norm_factor,
                   const bool is_odd) {
    return fft_c2r(out, in, norm_factor, is_odd, 2);
}

af_err af_fft3_c2r(af_array *out, const af_array in, const double norm_factor,
                   const bool is_odd) {
    return fft_c2r(out, in, norm_factor, is_odd, 3);
}

af_err af_set_fft_plan_cache_size(size_t cache_size) {
    try {
        detail::setFFTPlanCacheSize(cache_size);
    }
    CATCHALL;
    return AF_SUCCESS;
}
