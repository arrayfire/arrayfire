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
#include <fft.hpp>
#include <kernel/fft.hpp>
#include <platform.hpp>

#include <af/dim4.hpp>

using af::dim4;

namespace cpu {

void setFFTPlanCacheSize(size_t numPlans) { UNUSED(numPlans); }

template<typename T, int rank, bool direction>
void fft_inplace(Array<T> &in) {
    in.eval();
    getQueue().enqueue(kernel::fft_inplace<T, rank, direction>, in,
                       in.getDataDims());
}

template<typename Tc, typename Tr, int rank>
Array<Tc> fft_r2c(const Array<Tr> &in) {
    in.eval();

    dim4 odims    = in.dims();
    odims[0]      = odims[0] / 2 + 1;
    Array<Tc> out = createEmptyArray<Tc>(odims);

    getQueue().enqueue(kernel::fft_r2c<Tc, Tr, rank>, out, out.getDataDims(),
                       in, in.getDataDims());

    return out;
}

template<typename Tr, typename Tc, int rank>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims) {
    Array<Tr> out = createEmptyArray<Tr>(odims);

#ifdef USE_MKL
    getQueue().enqueue(kernel::fft_c2r<Tr, Tc, rank>, out, out.getDataDims(),
                       in, in.getDataDims(), odims);
#else
    if (rank > 1 || odims.ndims() > 1) {
        // FFTW does not have a input preserving algorithm for multidimensional
        // c2r FFTs
        Array<Tc> in_ = copyArray<Tc>(in);
        getQueue().enqueue(kernel::fft_c2r<Tr, Tc, rank>, out, out.getDataDims(),
                          in_, in.getDataDims(), odims);
    } else {
        getQueue().enqueue(kernel::fft_c2r<Tr, Tc, rank>, out, out.getDataDims(),
                          in, in.getDataDims(), odims);
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
