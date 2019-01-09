/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <copy.hpp>
#include <cufft.hpp>
#include <debug_cuda.hpp>
#include <fft.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <af/dim4.hpp>

using af::dim4;
using std::string;

namespace cuda {
void setFFTPlanCacheSize(size_t numPlans) {
    fftManager().setMaxCacheSize(numPlans);
}

template <typename T>
struct cufft_transform;

#define CUFFT_FUNC(T, TRANSFORM_TYPE)                                      \
    template <>                                                            \
    struct cufft_transform<T> {                                            \
        enum { type = CUFFT_##TRANSFORM_TYPE };                            \
        cufftResult operator()(cufftHandle plan, T *in, T *out, int dir) { \
            return cufftExec##TRANSFORM_TYPE(plan, in, out, dir);          \
        }                                                                  \
    };

CUFFT_FUNC(cfloat, C2C)
CUFFT_FUNC(cdouble, Z2Z)

template <typename To, typename Ti>
struct cufft_real_transform;

#define CUFFT_REAL_FUNC(To, Ti, TRANSFORM_TYPE)                     \
    template <>                                                     \
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

template <int rank>
void computeDims(int rdims[rank], const dim4 &idims) {
    for (int i = 0; i < rank; i++) { rdims[i] = idims[(rank - 1) - i]; }
}

template <typename T, int rank, bool direction>
void fft_inplace(Array<T> &in) {
    const dim4 idims    = in.dims();
    const dim4 istrides = in.strides();

    int t_dims[rank];
    int in_embed[rank];

    computeDims<rank>(t_dims, idims);
    computeDims<rank>(in_embed, in.getDataDims());

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= idims[i]; }

    SharedPlan plan =
        findPlan(rank, t_dims, in_embed, istrides[0], istrides[rank], in_embed,
                 istrides[0], istrides[rank],
                 (cufftType)cufft_transform<T>::type, batch);

    cufft_transform<T> transform;
    CUFFT_CHECK(cufftSetStream(*plan.get(), cuda::getActiveStream()));
    CUFFT_CHECK(transform(*plan.get(), (T *)in.get(), in.get(),
                          direction ? CUFFT_FORWARD : CUFFT_INVERSE));
}

template <typename Tc, typename Tr, int rank>
Array<Tc> fft_r2c(const Array<Tr> &in) {
    dim4 idims = in.dims();
    dim4 odims = in.dims();

    odims[0] = odims[0] / 2 + 1;

    Array<Tc> out = createEmptyArray<Tc>(odims);

    int t_dims[rank];
    int in_embed[rank], out_embed[rank];

    computeDims<rank>(t_dims, idims);
    computeDims<rank>(in_embed, in.getDataDims());
    computeDims<rank>(out_embed, out.getDataDims());

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= idims[i]; }

    dim4 istrides = in.strides();
    dim4 ostrides = out.strides();

    SharedPlan plan =
        findPlan(rank, t_dims, in_embed, istrides[0], istrides[rank], out_embed,
                 ostrides[0], ostrides[rank],
                 (cufftType)cufft_real_transform<Tc, Tr>::type, batch);

    cufft_real_transform<Tc, Tr> transform;
    CUFFT_CHECK(cufftSetStream(*plan.get(), cuda::getActiveStream()));
    CUFFT_CHECK(transform(*plan.get(), (Tr *)in.get(), out.get()));
    return out;
}

template <typename Tr, typename Tc, int rank>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims) {
    Array<Tr> out = createEmptyArray<Tr>(odims);

    int t_dims[rank];
    int in_embed[rank], out_embed[rank];

    computeDims<rank>(t_dims, odims);
    computeDims<rank>(in_embed, in.getDataDims());
    computeDims<rank>(out_embed, out.getDataDims());

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= odims[i]; }

    dim4 istrides = in.strides();
    dim4 ostrides = out.strides();

    cufft_real_transform<Tr, Tc> transform;

    SharedPlan plan =
        findPlan(rank, t_dims, in_embed, istrides[0], istrides[rank], out_embed,
                 ostrides[0], ostrides[rank],
                 (cufftType)cufft_real_transform<Tr, Tc>::type, batch);

    CUFFT_CHECK(cufftSetStream(*plan.get(), cuda::getActiveStream()));
    CUFFT_CHECK(transform(*plan.get(), (Tc *)in.get(), out.get()));
    return out;
}

#define INSTANTIATE(T)                                     \
    template void fft_inplace<T, 1, true>(Array<T> & in);  \
    template void fft_inplace<T, 2, true>(Array<T> & in);  \
    template void fft_inplace<T, 3, true>(Array<T> & in);  \
    template void fft_inplace<T, 1, false>(Array<T> & in); \
    template void fft_inplace<T, 2, false>(Array<T> & in); \
    template void fft_inplace<T, 3, false>(Array<T> & in);

INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

#define INSTANTIATE_REAL(Tr, Tc)                                \
    template Array<Tc> fft_r2c<Tc, Tr, 1>(const Array<Tr> &in); \
    template Array<Tc> fft_r2c<Tc, Tr, 2>(const Array<Tr> &in); \
    template Array<Tc> fft_r2c<Tc, Tr, 3>(const Array<Tr> &in); \
    template Array<Tr> fft_c2r<Tr, Tc, 1>(const Array<Tc> &in,  \
                                          const dim4 &odims);   \
    template Array<Tr> fft_c2r<Tr, Tc, 2>(const Array<Tc> &in,  \
                                          const dim4 &odims);   \
    template Array<Tr> fft_c2r<Tr, Tc, 3>(const Array<Tc> &in,  \
                                          const dim4 &odims);

INSTANTIATE_REAL(float, cfloat)
INSTANTIATE_REAL(double, cdouble)
}  // namespace cuda
