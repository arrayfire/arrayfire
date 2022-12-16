/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fft.hpp>

#include <Array.hpp>
#include <copy.hpp>
#include <cufft.hpp>
#include <debug_cuda.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <af/dim4.hpp>

#include <array>

using af::dim4;
using std::array;
using std::string;

namespace arrayfire {
namespace cuda {
void setFFTPlanCacheSize(size_t numPlans) {
    fftManager().setMaxCacheSize(numPlans);
}

template<typename T>
struct cufft_transform;

#define CUFFT_FUNC(T, TRANSFORM_TYPE)                                      \
    template<>                                                             \
    struct cufft_transform<T> {                                            \
        enum { type = CUFFT_##TRANSFORM_TYPE };                            \
        cufftResult operator()(cufftHandle plan, T *in, T *out, int dir) { \
            return cufftExec##TRANSFORM_TYPE(plan, in, out, dir);          \
        }                                                                  \
    };

CUFFT_FUNC(cfloat, C2C)
CUFFT_FUNC(cdouble, Z2Z)

template<typename To, typename Ti>
struct cufft_real_transform;

#define CUFFT_REAL_FUNC(To, Ti, TRANSFORM_TYPE)                     \
    template<>                                                      \
    struct cufft_real_transform<To, Ti> {                           \
        enum { type = CUFFT_##TRANSFORM_TYPE };                     \
        cufftResult operator()(cufftHandle plan, Ti *in, To *out) { \
            return cufftExec##TRANSFORM_TYPE(plan, in, out);        \
        }                                                           \
    };

CUFFT_REAL_FUNC(cfloat, float, R2C)
CUFFT_REAL_FUNC(cdouble, double, D2Z)

CUFFT_REAL_FUNC(float, cfloat, C2R)
CUFFT_REAL_FUNC(double, cdouble, Z2D)

inline array<int, AF_MAX_DIMS> computeDims(const int rank, const dim4 &idims) {
    array<int, AF_MAX_DIMS> retVal = {};
    for (int i = 0; i < rank; i++) { retVal[i] = idims[(rank - 1) - i]; }
    return retVal;
}

template<typename T>
void fft_inplace(Array<T> &in, const int rank, const bool direction) {
    const dim4 idims    = in.dims();
    const dim4 istrides = in.strides();

    auto t_dims   = computeDims(rank, idims);
    auto in_embed = computeDims(rank, in.getDataDims());

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= idims[i]; }

    SharedPlan plan =
        findPlan(rank, t_dims.data(), in_embed.data(), istrides[0],
                 istrides[rank], in_embed.data(), istrides[0], istrides[rank],
                 (cufftType)cufft_transform<T>::type, batch);

    cufft_transform<T> transform;
    CUFFT_CHECK(cufftSetStream(*plan.get(), getActiveStream()));
    CUFFT_CHECK(transform(*plan.get(), (T *)in.get(), in.get(),
                          direction ? CUFFT_FORWARD : CUFFT_INVERSE));
}

template<typename Tc, typename Tr>
Array<Tc> fft_r2c(const Array<Tr> &in, const int rank) {
    dim4 idims = in.dims();
    dim4 odims = in.dims();

    odims[0] = odims[0] / 2 + 1;

    Array<Tc> out = createEmptyArray<Tc>(odims);

    auto t_dims    = computeDims(rank, idims);
    auto in_embed  = computeDims(rank, in.getDataDims());
    auto out_embed = computeDims(rank, out.getDataDims());

    int batch = 1;
    for (int i = rank; i < AF_MAX_DIMS; i++) { batch *= idims[i]; }

    dim4 istrides = in.strides();
    dim4 ostrides = out.strides();

    SharedPlan plan =
        findPlan(rank, t_dims.data(), in_embed.data(), istrides[0],
                 istrides[rank], out_embed.data(), ostrides[0], ostrides[rank],
                 (cufftType)cufft_real_transform<Tc, Tr>::type, batch);

    cufft_real_transform<Tc, Tr> transform;
    CUFFT_CHECK(cufftSetStream(*plan.get(), getActiveStream()));
    CUFFT_CHECK(transform(*plan.get(), (Tr *)in.get(), out.get()));
    return out;
}

template<typename Tr, typename Tc>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims, const int rank) {
    Array<Tr> out = createEmptyArray<Tr>(odims);

    auto t_dims    = computeDims(rank, odims);
    auto in_embed  = computeDims(rank, in.getDataDims());
    auto out_embed = computeDims(rank, out.getDataDims());

    int batch = 1;
    for (int i = rank; i < AF_MAX_DIMS; i++) { batch *= odims[i]; }

    dim4 istrides = in.strides();
    dim4 ostrides = out.strides();

    cufft_real_transform<Tr, Tc> transform;

    SharedPlan plan =
        findPlan(rank, t_dims.data(), in_embed.data(), istrides[0],
                 istrides[rank], out_embed.data(), ostrides[0], ostrides[rank],
                 (cufftType)cufft_real_transform<Tr, Tc>::type, batch);

    CUFFT_CHECK(cufftSetStream(*plan.get(), getActiveStream()));
    CUFFT_CHECK(transform(*plan.get(), (Tc *)in.get(), out.get()));
    return out;
}

#define INSTANTIATE(T) \
    template void fft_inplace<T>(Array<T> &, const int, const bool);

INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

#define INSTANTIATE_REAL(Tr, Tc)                                               \
    template Array<Tc> fft_r2c<Tc, Tr>(const Array<Tr> &, const int);          \
    template Array<Tr> fft_c2r<Tr, Tc>(const Array<Tc> &in, const dim4 &odims, \
                                       const int);

INSTANTIATE_REAL(float, cfloat)
INSTANTIATE_REAL(double, cdouble)
}  // namespace cuda
}  // namespace arrayfire
