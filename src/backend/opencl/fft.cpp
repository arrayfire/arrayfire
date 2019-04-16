/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <clfft.hpp>
#include <copy.hpp>
#include <err_opencl.hpp>
#include <fft.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <af/dim4.hpp>

using af::dim4;
using std::string;

namespace opencl {

void setFFTPlanCacheSize(size_t numPlans) {
    fftManager().setMaxCacheSize(numPlans);
}

template<typename T>
struct Precision;
template<>
struct Precision<cfloat> {
    enum { type = CLFFT_SINGLE };
};
template<>
struct Precision<cdouble> {
    enum { type = CLFFT_DOUBLE };
};

static void computeDims(size_t rdims[4], const dim4 &idims) {
    for (int i = 0; i < 4; i++) { rdims[i] = (size_t)idims[i]; }
}

//(currently) true is in clFFT if length is a power of 2,3,5
inline bool isSupLen(dim_t length) {
    while (length > 1) {
        if (length % 2 == 0)
            length /= 2;
        else if (length % 3 == 0)
            length /= 3;
        else if (length % 5 == 0)
            length /= 5;
        else if (length % 7 == 0)
            length /= 7;
        else if (length % 11 == 0)
            length /= 11;
        else if (length % 13 == 0)
            length /= 13;
        else
            return false;
    }
    return true;
}

template<int rank>
void verifySupported(const dim4 dims) {
    for (int i = 0; i < rank; i++) { ARG_ASSERT(1, isSupLen(dims[i])); }
}

template<typename T, int rank, bool direction>
void fft_inplace(Array<T> &in) {
    verifySupported<rank>(in.dims());
    size_t tdims[4], istrides[4];

    computeDims(tdims, in.dims());
    computeDims(istrides, in.strides());

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= tdims[i]; }

    SharedPlan plan =
        findPlan(CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED,
                 (clfftDim)rank, tdims, istrides, istrides[rank], istrides,
                 istrides[rank], (clfftPrecision)Precision<T>::type, batch);

    cl_mem imem            = (*in.get())();
    cl_command_queue queue = getQueue()();

    CLFFT_CHECK(clfftEnqueueTransform(
        *plan.get(), direction ? CLFFT_FORWARD : CLFFT_BACKWARD, 1, &queue, 0,
        NULL, NULL, &imem, &imem, NULL));
}

template<typename Tc, typename Tr, int rank>
Array<Tc> fft_r2c(const Array<Tr> &in) {
    dim4 odims = in.dims();

    odims[0] = odims[0] / 2 + 1;

    Array<Tc> out = createEmptyArray<Tc>(odims);

    verifySupported<rank>(in.dims());
    size_t tdims[4], istrides[4], ostrides[4];

    computeDims(tdims, in.dims());
    computeDims(istrides, in.strides());
    computeDims(ostrides, out.strides());

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= tdims[i]; }

    SharedPlan plan =
        findPlan(CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED, (clfftDim)rank, tdims,
                 istrides, istrides[rank], ostrides, ostrides[rank],
                 (clfftPrecision)Precision<Tc>::type, batch);

    cl_mem imem            = (*in.get())();
    cl_mem omem            = (*out.get())();
    cl_command_queue queue = getQueue()();

    CLFFT_CHECK(clfftEnqueueTransform(*plan.get(), CLFFT_FORWARD, 1, &queue, 0,
                                      NULL, NULL, &imem, &omem, NULL));

    return out;
}

template<typename Tr, typename Tc, int rank>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims) {
    Array<Tr> out = createEmptyArray<Tr>(odims);

    verifySupported<rank>(odims);
    size_t tdims[4], istrides[4], ostrides[4];

    computeDims(tdims, odims);
    computeDims(istrides, in.strides());
    computeDims(ostrides, out.strides());

    int batch = 1;
    for (int i = rank; i < 4; i++) { batch *= tdims[i]; }

    SharedPlan plan =
        findPlan(CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL, (clfftDim)rank, tdims,
                 istrides, istrides[rank], ostrides, ostrides[rank],
                 (clfftPrecision)Precision<Tc>::type, batch);

    cl_mem imem            = (*in.get())();
    cl_mem omem            = (*out.get())();
    cl_command_queue queue = getQueue()();

    CLFFT_CHECK(clfftEnqueueTransform(*plan.get(), CLFFT_BACKWARD, 1, &queue, 0,
                                      NULL, NULL, &imem, &omem, NULL));

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
}  // namespace opencl
