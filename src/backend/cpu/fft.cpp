/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <fft.hpp>
#include <err_cpu.hpp>
#include <fftw3.h>
#include <copy.hpp>
#include <math.hpp>

using af::dim4;

namespace cpu
{

template<int rank>
void computeDims(int rdims[rank], const dim4 &idims)
{
    for (int i = 0; i < rank; i++) {
        rdims[i] = idims[(rank -1) - i];
    }
}

template<typename T>
struct fftw_transform;

#define TRANSFORM(PRE, TY)                                              \
    template<>                                                          \
    struct fftw_transform<TY>                                           \
    {                                                                   \
        typedef PRE##_plan plan_t;                                      \
        typedef PRE##_complex ctype_t;                                  \
                                                                        \
        template<typename... Args>                                      \
            plan_t create(Args... args)                                 \
        { return PRE##_plan_many_dft(args...); }                        \
        void execute(plan_t plan) { return PRE##_execute(plan); }       \
        void destroy(plan_t plan) { return PRE##_destroy_plan(plan); }  \
    };                                                                  \


TRANSFORM(fftwf, cfloat)
TRANSFORM(fftw, cdouble)

template<typename T, int rank, bool direction>
void fft_inplace(Array<T> &in)
{
    int t_dims[rank];
    int in_embed[rank];

    const dim4 idims = in.dims();

    computeDims<rank>(t_dims  , idims);
    computeDims<rank>(in_embed , in.getDataDims());

    const dim4 istrides = in.strides();

    typedef typename fftw_transform<T>::ctype_t ctype_t;
    typename fftw_transform<T>::plan_t plan;

    fftw_transform<T> transform;

    int batch = 1;
    for (int i = rank; i < 4; i++) {
        batch *= idims[i];
    }

    plan = transform.create(rank,
                            t_dims,
                            (int)batch,
                            (ctype_t *)in.get(),
                            in_embed, (int)istrides[0],
                            (int)istrides[rank],
                            (ctype_t *)in.get(),
                            in_embed, (int)istrides[0],
                            (int)istrides[rank],
                            direction ? FFTW_FORWARD : FFTW_BACKWARD,
                            FFTW_ESTIMATE);

    transform.execute(plan);
    transform.destroy(plan);
}

template<typename To, typename Ti>
struct fftw_real_transform;

#define TRANSFORM_REAL(PRE, To, Ti, POST)                               \
    template<>                                                          \
    struct fftw_real_transform<To, Ti>                                  \
    {                                                                   \
        typedef PRE##_plan plan_t;                                      \
        typedef PRE##_complex ctype_t;                                  \
                                                                        \
        template<typename... Args>                                      \
            plan_t create(Args... args)                                 \
        { return PRE##_plan_many_dft_##POST(args...); }                 \
        void execute(plan_t plan) { return PRE##_execute(plan); }       \
        void destroy(plan_t plan) { return PRE##_destroy_plan(plan); }  \
    };                                                                  \


TRANSFORM_REAL(fftwf, cfloat , float , r2c)
TRANSFORM_REAL(fftw , cdouble, double, r2c)
TRANSFORM_REAL(fftwf, float , cfloat , c2r)
TRANSFORM_REAL(fftw , double, cdouble, c2r)

template<typename Tc, typename Tr, int rank>
Array<Tc> fft_r2c(const Array<Tr> &in)
{
    dim4 idims = in.dims();
    dim4 odims = in.dims();

    odims[0] = odims[0] / 2 + 1;

    Array<Tc> out = createEmptyArray<Tc>(odims);

    int t_dims[rank];
    int in_embed[rank];
    int out_embed[rank];

    computeDims<rank>(t_dims  , idims);
    computeDims<rank>(in_embed , in.getDataDims());
    computeDims<rank>(out_embed , out.getDataDims());

    const dim4 istrides = in.strides();
    const dim4 ostrides = out.strides();

    typedef typename fftw_real_transform<Tc, Tr>::ctype_t ctype_t;
    typename fftw_real_transform<Tc, Tr>::plan_t plan;

    fftw_real_transform<Tc, Tr> transform;

    int batch = 1;
    for (int i = rank; i < 4; i++) {
        batch *= idims[i];
    }

    plan = transform.create(rank,
                            t_dims,
                            (int)batch,
                            (Tr *)in.get(),
                            in_embed, (int)istrides[0],
                            (int)istrides[rank],
                            (ctype_t *)out.get(),
                            out_embed, (int)ostrides[0],
                            (int)ostrides[rank],
                            FFTW_ESTIMATE);

    transform.execute(plan);
    transform.destroy(plan);

    return out;
}

template<typename Tr, typename Tc, int rank>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims)
{
    Array<Tr> out = createEmptyArray<Tr>(odims);

    int t_dims[rank];
    int in_embed[rank];
    int out_embed[rank];

    computeDims<rank>(t_dims  , odims);
    computeDims<rank>(in_embed , in.getDataDims());
    computeDims<rank>(out_embed , out.getDataDims());

    const dim4 istrides = in.strides();
    const dim4 ostrides = out.strides();

    typedef typename fftw_real_transform<Tr, Tc>::ctype_t ctype_t;
    typename fftw_real_transform<Tr, Tc>::plan_t plan;

    fftw_real_transform<Tr, Tc> transform;

    int batch = 1;
    for (int i = rank; i < 4; i++) {
        batch *= odims[i];
    }

    plan = transform.create(rank,
                            t_dims,
                            (int)batch,
                            (ctype_t *)in.get(),
                            in_embed, (int)istrides[0],
                            (int)istrides[rank],
                            (Tr *)out.get(),
                            out_embed, (int)ostrides[0],
                            (int)ostrides[rank],
                            FFTW_ESTIMATE);

    transform.execute(plan);
    transform.destroy(plan);
    return out;
}

#define INSTANTIATE(T)                                      \
    template void fft_inplace<T, 1, true >(Array<T> &in);   \
    template void fft_inplace<T, 2, true >(Array<T> &in);   \
    template void fft_inplace<T, 3, true >(Array<T> &in);   \
    template void fft_inplace<T, 1, false>(Array<T> &in);   \
    template void fft_inplace<T, 2, false>(Array<T> &in);   \
    template void fft_inplace<T, 3, false>(Array<T> &in);

    INSTANTIATE(cfloat )
    INSTANTIATE(cdouble)

#define INSTANTIATE_REAL(Tr, Tc)                                        \
    template Array<Tc> fft_r2c<Tc, Tr, 1>(const Array<Tr> &in);         \
    template Array<Tc> fft_r2c<Tc, Tr, 2>(const Array<Tr> &in);         \
    template Array<Tc> fft_r2c<Tc, Tr, 3>(const Array<Tr> &in);         \
    template Array<Tr> fft_c2r<Tr, Tc, 1>(const Array<Tc> &in, const dim4 &odims); \
    template Array<Tr> fft_c2r<Tr, Tc, 2>(const Array<Tc> &in, const dim4 &odims); \
    template Array<Tr> fft_c2r<Tr, Tc, 3>(const Array<Tc> &in, const dim4 &odims); \

    INSTANTIATE_REAL(float , cfloat )
    INSTANTIATE_REAL(double, cdouble)
}
