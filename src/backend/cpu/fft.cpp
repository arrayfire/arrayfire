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
void computeDims(int *rdims, const dim4 &idims)
{
    if (rank==3) {
        rdims[0] = idims[2];
        rdims[1] = idims[1];
        rdims[2] = idims[0];
    } else if(rank==2) {
        rdims[0] = idims[1];
        rdims[1] = idims[0];
    } else {
        rdims[0] = idims[0];
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
    int in_dims[3];
    int in_embed[3];
    int out_embed[3];

    const dim4 idims = in.dims();

    computeDims<rank>(in_dims  , idims);
    computeDims<rank>(in_embed , in.getDataDims());
    computeDims<rank>(out_embed, out.getDataDims());

    const dim4 istrides = in.strides();
    const dim4 ostrides = out.strides();

    typedef typename fftw_transform<T>::ctype_t ctype_t;
    typename fftw_transform<T>::plan_t plan;

    fftw_transform<T> transform;

    plan = transform.create(rank,
                            in_dims,
                            (int)idims[rank],
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

template<int rank>
void computePaddedDims(dim4 &pdims, dim_type const * const pad)
{
    if (rank==1) {
        pdims[0] = pad[0];
    } else if (rank==2) {
        pdims[0] = pad[0];
        pdims[1] = pad[1];
    } else if (rank==3) {
        pdims[0] = pad[0];
        pdims[1] = pad[1];
        pdims[2] = pad[2];
    }
}

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType> fft(Array<inType> const &in, double norm_factor, dim_type const npad, dim_type const * const pad)
{
    ARG_ASSERT(1, ((in.isOwner()==true) && "fft: Sub-Arrays not supported yet."));
    ARG_ASSERT(1, rank >= 1 && rank <= 3);

    dim4 pdims(1);
    computePaddedDims<rank>(pdims, pad);
    pdims[rank] = in.dims()[rank];

    Array<outType> ret = padArray<inType, outType>(in, (npad>0 ? pdims : in.dims()));
    fft_common<outType, rank, true>(ret, ret);
    return ret;
}

template<typename T, int rank>
Array<T> ifft(Array<T> const &in, double norm_factor, dim_type const npad, dim_type const * const pad)
{
    ARG_ASSERT(1, ((in.isOwner()==true) && "ifft: Sub-Arrays not supported yet."));
    ARG_ASSERT(1, rank >= 1 && rank <= 3);

    dim4 pdims(1);
    computePaddedDims<rank>(pdims, pad);
    pdims[rank] = in.dims()[rank];

    Array<T> ret = padArray<T, T>(in, (npad>0 ? pdims : in.dims()), scalar<T>(0), norm_factor);
    fft_common<T, rank, false>(ret, ret);

    return ret;
}

#define INSTANTIATE1(T1, T2)\
    template Array<T2> fft <T1, T2, 1, true >(const Array<T1> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T2> fft <T1, T2, 2, true >(const Array<T1> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T2> fft <T1, T2, 3, true >(const Array<T1> &in, double norm_factor, dim_type const npad, dim_type const * const pad);

INSTANTIATE1(float  , cfloat )
INSTANTIATE1(double , cdouble)

#define INSTANTIATE2(T)\
    template Array<T> fft <T, T, 1, false>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T> fft <T, T, 2, false>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T> fft <T, T, 3, false>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T> ifft<T, 1>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T> ifft<T, 2>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T> ifft<T, 3>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad);

INSTANTIATE2(cfloat )
INSTANTIATE2(cdouble)

}
