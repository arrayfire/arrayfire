/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <cassert>
#include <cstring>

namespace cpu
{
namespace kernel
{

enum PadDirection {
    PRE,
    POST,
    BOTH
};

void getPadInfo(af::dim4& outOffsets, af::dim4& outDims, af::dim4 const & inDims, const af::dim4 padSizes, const PadDirection direction[4]) {
    for (int n = 0; n < 4; n++) {
        switch (direction[n]) {
        case PRE:
            outOffsets[n] = padSizes[n];
            outDims[n] = inDims[n] + padSizes[n];
            break;
        case POST:
            outOffsets[n] = 0;
            outDims[n] = inDims[n] + padSizes[n];
            break;
        case BOTH:
            outOffsets[n] = padSizes[n];
            outDims[n] = inDims[n] + 2*padSizes[n];
            break;
        }
    }
}

template<typename T>
void padArray(T * outData, af::dim4 const & outStrides, af::dim4 const & outOffsets, const T * inData, af::dim4 const & inDims, af::dim4 const & inStrides) {
    for (dim_t l = 0; l < inDims[3]; ++l) {
        for (dim_t k = 0; k < inDims[2]; ++k) {
            for (dim_t j = 0; j < inDims[1]; ++j) {
                dim_t outOffset = getIdx(outStrides, outOffsets[0], j + outOffsets[1], k + outOffsets[2], l + outOffsets[3]);
                dim_t inOffset = getIdx(inStrides, 0, j, k, l);
                memcpy(outData + outOffset, inData + inOffset, sizeof(T) * inDims[0]);
            }
        }
    }
}

template<typename T>
void padArray(Array<T>& out, const Array<T>& in, const af::dim4 padSizes, const PadDirection direction[4]) {
    af::dim4 outOffsets, outDims;
    getPadInfo(outOffsets, outDims, in.dims(), padSizes, direction);
    assert(outDims[0] == out.dims()[0]);
    assert(outDims[1] == out.dims()[1]);
    assert(outDims[2] == out.dims()[2]);
    assert(outDims[3] == out.dims()[3]);

    padArray(out.get(), out.strides(), outOffsets, in.get(), in.dims(), in.strides());
}

template<typename T>
Array<T> createPaddedArray(const Array<T>& in, const af::dim4 padSizes, T padValue, const PadDirection direction[4]) {
    af::dim4 outOffsets, outDims;
    getPadInfo(outOffsets, outDims, in.dims(), padSizes, direction);

    Array<T> out = createEmptyArray<T>(outDims);
    std::fill(out.get(), out.get() + out.elements(), padValue);
    padArray(out, in, padSizes, direction);
    return out;
}

void getCropInfo(af::dim4& inOffsets, af::dim4& outDims, af::dim4 const & inDims, const af::dim4 cropSizes, const PadDirection direction[4]) {
    for (int n = 0; n < 4; n++) {
        switch (direction[n]) {
        case PRE:
            inOffsets[n] = cropSizes[n];
            outDims[n] = inDims[n] - cropSizes[n];
            break;
        case POST:
            inOffsets[n] = 0;
            outDims[n] = inDims[n] - cropSizes[n];
            break;
        case BOTH:
            inOffsets[n] = cropSizes[n];
            outDims[n] = inDims[n] - 2*cropSizes[n];
            break;
        }
    }
}

template<typename T>
void cropArray(T * outData, af::dim4 const & outDims, af::dim4 const & outStrides, const T * inData, af::dim4 const & inOffsets, af::dim4 const & inStrides) {
    for (dim_t l = 0; l < outDims[3]; ++l) {
        for (dim_t k = 0; k < outDims[2]; ++k) {
            for (dim_t j = 0; j < outDims[1]; ++j) {
                dim_t inOffset = getIdx(inStrides, inOffsets[0], j + inOffsets[1], k + inOffsets[2], l + inOffsets[3]);
                dim_t outOffset = getIdx(outStrides, 0, j, k, l);
                memcpy(outData + outOffset, inData + inOffset, sizeof(T) * outDims[0]);
            }
        }
    }
}

template<typename T>
void cropArray(Array<T>& out, const Array<T>& in, const af::dim4 cropSizes, const PadDirection direction[4]) {
    af::dim4 inOffsets, outDims;
    getCropInfo(inOffsets, outDims, in.dims(), cropSizes, direction);
    assert(outDims[0] == out.dims()[0]);
    assert(outDims[1] == out.dims()[1]);
    assert(outDims[2] == out.dims()[2]);
    assert(outDims[3] == out.dims()[3]);

    cropArray(out.get(), outDims, out.strides(), in.get(), inOffsets, in.strides());
}

template<typename T>
Array<T> createCroppedArray(const Array<T>& in, const af::dim4 cropSizes, const PadDirection direction[4]) {
    af::dim4 inOffsets, outDims;
    getCropInfo(inOffsets, outDims, in.dims(), cropSizes, direction);

    Array<T> out = createEmptyArray<T>(outDims);
    cropArray(out, in, cropSizes, direction);
    return out;
}

}
}
