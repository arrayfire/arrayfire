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

template<typename T, int rank, int direction>
void fft_common(Array <T> &out, const Array<T> &in)
{
    int in_dims[rank];
    int in_embed[rank];
    int out_embed[rank];

    const dim4 idims = in.dims();

    computeDims<rank>(in_dims  , idims);
    computeDims<rank>(in_embed , in.getDataDims());
    computeDims<rank>(out_embed, out.getDataDims());

    const dim4 istrides = in.strides();
    const dim4 ostrides = out.strides();

    typedef typename fftw_transform<T>::ctype_t ctype_t;
    typename fftw_transform<T>::plan_t plan;

    fftw_transform<T> transform;

    int batch = 1;
    for (int i = rank; i < 4; i++) {
        batch *= idims[i];
    }

    plan = transform.create(rank,
                            in_dims,
                            (int)batch,
                            (ctype_t *)in.get(),
                            in_embed, (int)istrides[0],
                            (int)istrides[rank],
                            (ctype_t *)out.get(),
                            out_embed, (int)ostrides[0],
                            (int)ostrides[rank],
                            direction ? FFTW_FORWARD : FFTW_BACKWARD,
                            FFTW_ESTIMATE);

    transform.execute(plan);
    transform.destroy(plan);

}

void computePaddedDims(dim4 &pdims,
                       const dim4 &idims,
                       const dim_t npad,
                       dim_t const * const pad)
{
    for (int i = 0; i < 4; i++) {
        pdims[i] = (i < (int)npad) ? pad[i] : idims[i];
    }
}

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType> fft(Array<inType> const &in, double norm_factor, dim_t const npad, dim_t const * const pad)
{
    ARG_ASSERT(1, rank >= 1 && rank <= 3);

    dim4 pdims(1);
    computePaddedDims(pdims, in.dims(), npad, pad);

    Array<outType> ret = padArray<inType, outType>(in, pdims);
    fft_common<outType, rank, true>(ret, ret);
    return ret;
}

template<typename T, int rank>
Array<T> ifft(Array<T> const &in, double norm_factor, dim_t const npad, dim_t const * const pad)
{
    ARG_ASSERT(1, rank >= 1 && rank <= 3);

    dim4 pdims(1);
    computePaddedDims(pdims, in.dims(), npad, pad);

    Array<T> ret = padArray<T, T>(in, pdims, scalar<T>(0), norm_factor);
    fft_common<T, rank, false>(ret, ret);

    return ret;
}

#define INSTANTIATE1(T1, T2)\
    template Array<T2> fft <T1, T2, 1, true >(const Array<T1> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T2> fft <T1, T2, 2, true >(const Array<T1> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T2> fft <T1, T2, 3, true >(const Array<T1> &in, double norm_factor, dim_t const npad, dim_t const * const pad);

INSTANTIATE1(float  , cfloat )
INSTANTIATE1(double , cdouble)

#define INSTANTIATE2(T)\
    template Array<T> fft <T, T, 1, false>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T> fft <T, T, 2, false>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T> fft <T, T, 3, false>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T> ifft<T, 1>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T> ifft<T, 2>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T> ifft<T, 3>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad);

INSTANTIATE2(cfloat )
INSTANTIATE2(cdouble)

}
