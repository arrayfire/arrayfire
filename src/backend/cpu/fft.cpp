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

#define TRANSFORM(T, PREFIX)                            \
    template<int rank, int direction>                   \
    void fftw_common(Array<T> &arr)                     \
    {                                                   \
        int rank_dims[3];                               \
        const dim4 dims = arr.dims();                   \
        computeDims<rank>(rank_dims, dims);             \
        const dim4 strides = arr.strides();             \
        PREFIX##_plan plan = PREFIX##_plan_many_dft  (  \
            rank,                                       \
            rank_dims,                                  \
            (int)dims[rank],                            \
            (PREFIX##_complex *)arr.get(),              \
            NULL, (int)strides[0],                      \
            (int)strides[rank],                         \
            (PREFIX##_complex *)arr.get(),              \
            NULL, (int)strides[0],                      \
            (int)strides[rank],                         \
            direction,                                  \
            FFTW_ESTIMATE);                             \
        PREFIX##_execute(plan);                         \
        PREFIX##_destroy_plan(plan);                    \
    }

TRANSFORM(cfloat , fftwf);
TRANSFORM(cdouble, fftw );

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
    fftw_common<rank, FFTW_FORWARD>(ret);
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
    fftw_common<rank, FFTW_BACKWARD>(ret);

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
