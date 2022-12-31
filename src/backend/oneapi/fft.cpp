/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fft.hpp>

#include <copy.hpp>
#include <err_oneapi.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace oneapi {

void setFFTPlanCacheSize(size_t numPlans) {}

/*
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
*/

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
    ONEAPI_NOT_SUPPORTED("");
}

template<typename Tc, typename Tr>
Array<Tc> fft_r2c(const Array<Tr> &in, const int rank) {
    ONEAPI_NOT_SUPPORTED("");
    dim4 odims = in.dims();

    odims[0] = odims[0] / 2 + 1;

    Array<Tc> out = createEmptyArray<Tc>(odims);
    return out;
}

template<typename Tr, typename Tc>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims, const int rank) {
    ONEAPI_NOT_SUPPORTED("");
    Array<Tr> out = createEmptyArray<Tr>(odims);
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
}  // namespace oneapi
}  // namespace arrayfire
