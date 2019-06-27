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
#include <fftw3.h>
#include <platform.hpp>
#include <types.hpp>
#include <af/dim4.hpp>

#include <type_traits>

using af::dim4;

namespace cpu {

template<typename T>
struct fftw_transform;

#define TRANSFORM(PRE, TY)                                             \
    template<>                                                         \
    struct fftw_transform<TY> {                                        \
        typedef PRE##_plan plan_t;                                     \
        typedef PRE##_complex ctype_t;                                 \
                                                                       \
        template<typename... Args>                                     \
        plan_t create(Args... args) {                                  \
            return PRE##_plan_many_dft(args...);                       \
        }                                                              \
        void execute(plan_t plan) { return PRE##_execute(plan); }      \
        void destroy(plan_t plan) { return PRE##_destroy_plan(plan); } \
    };

TRANSFORM(fftwf, cfloat)
TRANSFORM(fftw, cdouble)

template<typename To, typename Ti>
struct fftw_real_transform;

#define TRANSFORM_REAL(PRE, To, Ti, POST)                              \
    template<>                                                         \
    struct fftw_real_transform<To, Ti> {                               \
        typedef PRE##_plan plan_t;                                     \
        typedef PRE##_complex ctype_t;                                 \
                                                                       \
        template<typename... Args>                                     \
        plan_t create(Args... args) {                                  \
            return PRE##_plan_many_dft_##POST(args...);                \
        }                                                              \
        void execute(plan_t plan) { return PRE##_execute(plan); }      \
        void destroy(plan_t plan) { return PRE##_destroy_plan(plan); } \
    };

TRANSFORM_REAL(fftwf, cfloat, float, r2c)
TRANSFORM_REAL(fftw, cdouble, double, r2c)
TRANSFORM_REAL(fftwf, float, cfloat, c2r)
TRANSFORM_REAL(fftw, double, cdouble, c2r)

template<int rank>
void computeDims(int rdims[rank], const af::dim4 &idims) {
    for (int i = 0; i < rank; i++) { rdims[i] = idims[(rank - 1) - i]; }
}

void setFFTPlanCacheSize(size_t numPlans) { UNUSED(numPlans); }

template<typename T, int rank, bool direction>
void fft_inplace(Array<T> &in) {
    auto func = [=](Param<T> in, const af::dim4 iDataDims) {
        int t_dims[rank];
        int in_embed[rank];

        const af::dim4 idims = in.dims();

        computeDims<rank>(t_dims, idims);
        computeDims<rank>(in_embed, iDataDims);

        const af::dim4 istrides = in.strides();

        typedef typename fftw_transform<T>::ctype_t ctype_t;
        typename fftw_transform<T>::plan_t plan;

        fftw_transform<T> transform;

        int batch = 1;
        for (int i = rank; i < 4; i++) { batch *= idims[i]; }

        plan = transform.create(
            rank, t_dims, (int)batch, (ctype_t *)in.get(), in_embed,
            (int)istrides[0], (int)istrides[rank], (ctype_t *)in.get(),
            in_embed, (int)istrides[0], (int)istrides[rank],
            direction ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_ESTIMATE);

        transform.execute(plan);
        transform.destroy(plan);
    };
    getQueue().enqueue(func, in, in.getDataDims());
}

template<typename Tc, typename Tr, int rank>
Array<Tc> fft_r2c(const Array<Tr> &in) {
    dim4 odims    = in.dims();
    odims[0]      = odims[0] / 2 + 1;
    Array<Tc> out = createEmptyArray<Tc>(odims);

    auto func = [=](Param<Tc> out, const af::dim4 oDataDims, CParam<Tr> in,
                    const af::dim4 iDataDims) {
        af::dim4 idims = in.dims();

        int t_dims[rank];
        int in_embed[rank];
        int out_embed[rank];

        computeDims<rank>(t_dims, idims);
        computeDims<rank>(in_embed, iDataDims);
        computeDims<rank>(out_embed, oDataDims);

        const af::dim4 istrides = in.strides();
        const af::dim4 ostrides = out.strides();

        typedef typename fftw_real_transform<Tc, Tr>::ctype_t ctype_t;
        typename fftw_real_transform<Tc, Tr>::plan_t plan;

        fftw_real_transform<Tc, Tr> transform;

        int batch = 1;
        for (int i = rank; i < 4; i++) { batch *= idims[i]; }

        plan = transform.create(
            rank, t_dims, (int)batch, (Tr *)in.get(), in_embed,
            (int)istrides[0], (int)istrides[rank], (ctype_t *)out.get(),
            out_embed, (int)ostrides[0], (int)ostrides[rank], FFTW_ESTIMATE);

        transform.execute(plan);
        transform.destroy(plan);
    };

    getQueue().enqueue(func, out, out.getDataDims(), in, in.getDataDims());

    return out;
}

template<typename Tr, typename Tc, int rank>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims) {
    Array<Tr> out = createEmptyArray<Tr>(odims);

    auto func = [=](Param<Tr> out, const af::dim4 oDataDims, CParam<Tc> in,
                    const af::dim4 iDataDims, const af::dim4 odims) {
        int t_dims[rank];
        int in_embed[rank];
        int out_embed[rank];

        computeDims<rank>(t_dims, odims);
        computeDims<rank>(in_embed, iDataDims);
        computeDims<rank>(out_embed, oDataDims);

        const af::dim4 istrides = in.strides();
        const af::dim4 ostrides = out.strides();

        typedef typename fftw_real_transform<Tr, Tc>::ctype_t ctype_t;
        typename fftw_real_transform<Tr, Tc>::plan_t plan;

        fftw_real_transform<Tr, Tc> transform;

        int batch = 1;
        for (int i = rank; i < 4; i++) { batch *= odims[i]; }

        // By default, fftw estimate flag is sufficient for most transforms.
        // However, complex to real transforms modify the input data memory
        // while performing the transformation. To avoid that, we need to pass
        // FFTW_PRESERVE_INPUT also. This flag however only works for 1D
        // transforms and for higher level transformations, a copy of input
        // data is passed onto the upstream FFTW calls.
        unsigned int flags = FFTW_ESTIMATE;
        if (rank == 1) { flags |= FFTW_PRESERVE_INPUT; }

        plan = transform.create(rank, t_dims, (int)batch, (ctype_t *)in.get(),
                                in_embed, (int)istrides[0], (int)istrides[rank],
                                (Tr *)out.get(), out_embed, (int)ostrides[0],
                                (int)ostrides[rank], flags);

        transform.execute(plan);
        transform.destroy(plan);
    };

#ifdef USE_MKL
    getQueue().enqueue(func, out, out.getDataDims(), in, in.getDataDims(),
                       odims);
#else
    if (rank > 1 || odims.ndims() > 1) {
        // FFTW does not have a input preserving algorithm for multidimensional
        // c2r FFTs
        Array<Tc> in_ = copyArray<Tc>(in);
        getQueue().enqueue(func, out, out.getDataDims(), in_, in.getDataDims(),
                           odims);
    } else {
        getQueue().enqueue(func, out, out.getDataDims(), in, in.getDataDims(),
                           odims);
    }
#endif

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

}  // namespace cpu
