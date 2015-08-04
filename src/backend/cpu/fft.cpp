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
    int in_dims[rank];
    int in_embed[rank];

    const dim4 idims = in.dims();

    computeDims<rank>(in_dims  , idims);
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
                            in_dims,
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

#define INSTANTIATE(T)                                      \
    template void fft_inplace<T, 1, true >(Array<T> &in);   \
    template void fft_inplace<T, 2, true >(Array<T> &in);   \
    template void fft_inplace<T, 3, true >(Array<T> &in);   \
    template void fft_inplace<T, 1, false>(Array<T> &in);   \
    template void fft_inplace<T, 2, false>(Array<T> &in);   \
    template void fft_inplace<T, 3, false>(Array<T> &in);

    INSTANTIATE(cfloat )
    INSTANTIATE(cdouble)


}
