/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fft.hpp>

#include <clfft.hpp>
#include <copy.hpp>
#include <err_opencl.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
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

void computeDims(size_t rdims[AF_MAX_DIMS], const dim4 &idims) {
    for (int i = 0; i < AF_MAX_DIMS; i++) {
        rdims[i] = static_cast<size_t>(idims[i]);
    }
}

//(currently) true is in clFFT if length is a power of 2,3,5
inline bool isSupLen(dim_t length) {
    while (length > 1) {
        if (length % 2 == 0) {
            length /= 2;
        } else if (length % 3 == 0) {
            length /= 3;
        } else if (length % 5 == 0) {
            length /= 5;
        } else if (length % 7 == 0) {
            length /= 7;
        } else if (length % 11 == 0) {
            length /= 11;
        } else if (length % 13 == 0) {
            length /= 13;
        } else {
            return false;
        }
    }
    return true;
}

void verifySupported(const int rank, const dim4 &dims) {
    for (int i = 0; i < rank; i++) { ARG_ASSERT(1, isSupLen(dims[i])); }
}

template<typename T>
void fft_inplace(Array<T> &in, const int rank, const bool direction) {
    verifySupported(rank, in.dims());
    size_t tdims[AF_MAX_DIMS], istrides[AF_MAX_DIMS];

    computeDims(tdims, in.dims());
    computeDims(istrides, in.strides());

    int batch = 1;
    for (int i = rank; i < AF_MAX_DIMS; i++) { batch *= tdims[i]; }

    SharedPlan plan = findPlan(
        CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED,
        static_cast<clfftDim>(rank), tdims, istrides, istrides[rank], istrides,
        istrides[rank], static_cast<clfftPrecision>(Precision<T>::type), batch);

    cl_mem imem            = (*in.get())();
    cl_command_queue queue = getQueue()();

    CLFFT_CHECK(clfftEnqueueTransform(
        *plan.get(), direction ? CLFFT_FORWARD : CLFFT_BACKWARD, 1, &queue, 0,
        NULL, NULL, &imem, &imem, NULL));
}

template<typename Tc, typename Tr>
Array<Tc> fft_r2c(const Array<Tr> &in, const int rank) {
    dim4 odims = in.dims();

    odims[0] = odims[0] / 2 + 1;

    Array<Tc> out = createEmptyArray<Tc>(odims);

    verifySupported(rank, in.dims());
    size_t tdims[AF_MAX_DIMS], istrides[AF_MAX_DIMS], ostrides[AF_MAX_DIMS];

    computeDims(tdims, in.dims());
    computeDims(istrides, in.strides());
    computeDims(ostrides, out.strides());

    int batch = 1;
    for (int i = rank; i < AF_MAX_DIMS; i++) { batch *= tdims[i]; }

    SharedPlan plan = findPlan(
        CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED, static_cast<clfftDim>(rank),
        tdims, istrides, istrides[rank], ostrides, ostrides[rank],
        static_cast<clfftPrecision>(Precision<Tc>::type), batch);

    cl_mem imem            = (*in.get())();
    cl_mem omem            = (*out.get())();
    cl_command_queue queue = getQueue()();

    CLFFT_CHECK(clfftEnqueueTransform(*plan.get(), CLFFT_FORWARD, 1, &queue, 0,
                                      NULL, NULL, &imem, &omem, NULL));

    return out;
}

template<typename Tr, typename Tc>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims, const int rank) {
    Array<Tr> out = createEmptyArray<Tr>(odims);

    verifySupported(rank, odims);
    size_t tdims[AF_MAX_DIMS], istrides[AF_MAX_DIMS], ostrides[AF_MAX_DIMS];

    computeDims(tdims, odims);
    computeDims(istrides, in.strides());
    computeDims(ostrides, out.strides());

    int batch = 1;
    for (int i = rank; i < AF_MAX_DIMS; i++) { batch *= tdims[i]; }

    SharedPlan plan = findPlan(
        CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL, static_cast<clfftDim>(rank),
        tdims, istrides, istrides[rank], ostrides, ostrides[rank],
        static_cast<clfftPrecision>(Precision<Tc>::type), batch);

    cl_mem imem            = (*in.get())();
    cl_mem omem            = (*out.get())();
    cl_command_queue queue = getQueue()();

    CLFFT_CHECK(clfftEnqueueTransform(*plan.get(), CLFFT_BACKWARD, 1, &queue, 0,
                                      NULL, NULL, &imem, &omem, NULL));

    return out;
}

#define INSTANTIATE(T) \
    template void fft_inplace<T>(Array<T> &, const int, const bool);

INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

#define INSTANTIATE_REAL(Tr, Tc)                                        \
    template Array<Tc> fft_r2c<Tc, Tr>(const Array<Tr> &, const int);   \
    template Array<Tr> fft_c2r<Tr, Tc>(const Array<Tc> &, const dim4 &, \
                                       const int);

INSTANTIATE_REAL(float, cfloat)
INSTANTIATE_REAL(double, cdouble)
}  // namespace opencl
}  // namespace arrayfire
