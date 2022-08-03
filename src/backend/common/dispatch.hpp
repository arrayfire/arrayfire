/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <assert.h>
#include <platform.hpp>
#include <af/defines.h>
#include <algorithm>
#include <cmath>

#define divup(a, b) (((a) + (b)-1) / (b))

unsigned nextpow2(unsigned x);

// isPrime & greatestPrimeFactor are tailored after
// itk::Math::{IsPrimt, GreatestPrimeFactor}
template<typename T>
inline bool isPrime(T n) {
    if (n <= 1) return false;

    const T last{(T)std::sqrt((double)n)};
    for (T x{2}; x <= last; ++x) {
        if (n % x == 0) return false;
    }

    return true;
}

template<typename T>
inline T greatestPrimeFactor(T n) {
    T v{2};

    while (v <= n) {
        if (n % v == 0 && isPrime(v))
            n /= v;
        else
            v += 1;
    }

    return v;
}
// Empty columns (dim==1) in refDims are removed from dims & strides.
// INPUT: refDims, refNdims
// UPDATE: dims, strides
// RETURN: ndims
template<typename T>
T removeEmptyColumns(const T refDims[AF_MAX_DIMS], const T refNdims,
                     T dims[AF_MAX_DIMS], T strides[AF_MAX_DIMS]) {
    T ndims{0};
    const T* refPtr{refDims};
    const T* refPtr_end{refDims + refNdims};
    // Search for first dimension == 1
    while (refPtr != refPtr_end && *refPtr != 1) {
        ++refPtr;
        ++ndims;
    }
    if (ndims != refNdims) {
        T* dPtr_out{dims + ndims};
        const T* dPtr_in{dPtr_out};
        T* sPtr_out{strides + ndims};
        const T* sPtr_in{sPtr_out};
        // Compress all remaining dimensions
        while (refPtr != refPtr_end) {
            if (*refPtr != 1) {
                *(dPtr_out++) = *dPtr_in;
                *(sPtr_out++) = *sPtr_in;
                ++ndims;
            }
            ++refPtr;
            ++dPtr_in;
            ++sPtr_in;
        }
        // Fill remaining dimensions with 1 and calculate corresponding strides
        // lastStride = last written dim * last written stride
        const T lastStride{*(dPtr_out - 1) * *(sPtr_out - 1)};
        const T lastDim{1};
        for (const T* dPtr_end{dims + AF_MAX_DIMS}; dPtr_out != dPtr_end;
             ++dPtr_out, ++sPtr_out) {
            *dPtr_out = lastDim;
            *sPtr_out = lastStride;
        }
    }
    return ndims;
}

// Empty columns (dim==1) in refDims are removed from strides
// ASSUMPTION: dims are equal to refDims, so are not provided
// INPUT: refDims, refNdims
// UPDATE: strides
// RETURN: ndims
template<typename T>
T removeEmptyColumns(const T refDims[AF_MAX_DIMS], const T refNdims,
                     T strides[AF_MAX_DIMS]) {
    T ndims{0};
    const T* refPtr{refDims};
    const T* refPtr_end{refDims + refNdims};
    // Search for first dimension == 1
    while (refPtr != refPtr_end && *refPtr != 1) {
        ++refPtr;
        ++ndims;
    }
    if (ndims != refNdims) {
        T* sPtr_out{strides + ndims};
        const T* sPtr_in{sPtr_out};
        // Compress all remaining dimensions
        while (refPtr != refPtr_end) {
            if (*refPtr != 1) {
                *(sPtr_out++) = *sPtr_in;
                ++ndims;
            };
            ++refPtr;
            ++sPtr_in;
        }
        // Calculate remaining strides
        // lastStride = last written dim * last written stride
        const T lastStride{*(refPtr - 1) * *(sPtr_out - 1)};
        for (const T* sPtr_end{strides + AF_MAX_DIMS}; sPtr_out != sPtr_end;
             ++sPtr_out) {
            *sPtr_out = lastStride;
        }
    }
    return ndims;
}

// Columns with the same stride in both arrays are combined.  Both arrays will
// remain in sync and will return the same ndims.
// ASSUMPTION: both arrays have the same ndims
// UPDATE: dims1, strides1, UPDATE: dims2, strides2, ndims
// RETURN: ndims
template<typename T>
T combineColumns(T dims1[AF_MAX_DIMS], T strides1[AF_MAX_DIMS], T& ndims,
                 T dims2[AF_MAX_DIMS], T strides2[AF_MAX_DIMS]) {
    for (T c{0}; c < ndims - 1; ++c) {
        if (dims1[c] == dims2[c] && dims1[c] * strides1[c] == strides1[c + 1] &&
            dims1[c] * strides2[c] == strides2[c + 1]) {
            // Combine columns, since they are linear
            // This will increase the dimension of the resulting column,
            // given more opportunities for kernel optimization
            dims1[c] *= dims1[c + 1];
            dims2[c] *= dims2[c + 1];
            --ndims;
            for (T i{c + 1}; i < ndims; ++i) {
                dims1[i]    = dims1[i + 1];
                dims2[i]    = dims2[i + 1];
                strides1[i] = strides1[i + 1];
                strides2[i] = strides2[i + 1];
            }
            dims1[ndims] = 1;
            dims2[ndims] = 1;
            --c;  // Redo this colum, since it is removed now
        }
    }
    return ndims;
}
// Columns with the same stride in both arrays are combined.  Both arrays will
// remain in sync and will return the same ndims.
// ASSUMPTION: both arrays have the same dims
// UPDATE: dims1, strides1,
// UPDATE: strides2, ndims
// RETURN: ndims
template<typename T>
T combineColumns(T dims1[AF_MAX_DIMS], T strides1[AF_MAX_DIMS], T& ndims,
                 T strides2[AF_MAX_DIMS]) {
    for (T c{0}; c < ndims - 1; ++c) {
        if (dims1[c] * strides1[c] == strides1[c + 1] &&
            dims1[c] * strides2[c] == strides2[c + 1]) {
            // Combine columns, since they are linear
            // This will increase the dimension of the resulting column,
            // given more opportunities for kernel optimization
            dims1[c] *= dims1[c + 1];
            --ndims;
            for (T i{c + 1}; i < ndims; ++i) {
                dims1[i]    = dims1[i + 1];
                strides1[i] = strides1[i + 1];
                strides2[i] = strides2[i + 1];
            }
            dims1[ndims] = 1;
            --c;  // Redo this colum, since it is removed now
        }
    }
    return ndims;
}