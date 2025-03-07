/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/internal_enums.hpp>

namespace arrayfire {
namespace cuda {

template<typename To, typename Ti>
__global__ void packData(Param<To> out, CParam<Ti> in, const int di0_half,
                         const bool odd_di0) {
    const int t = blockDim.x * blockIdx.x + threadIdx.x;

    const int tMax = out.strides[3] * out.dims[3];

    if (t >= tMax) return;

    const int do1 = out.dims[1];
    const int do2 = out.dims[2];
    const int so1 = out.strides[1];
    const int so2 = out.strides[2];
    const int so3 = out.strides[3];

    const int to0 = t % so1;
    const int to1 = (t / so1) % do1;
    const int to2 = (t / so2) % do2;
    const int to3 = t / so3;

    const int di1 = in.dims[1];
    const int di2 = in.dims[2];
    const int si1 = in.strides[1];
    const int si2 = in.strides[2];
    const int si3 = in.strides[3];

    const int ti0 = to0;
    const int ti1 = to1 * si1;
    const int ti2 = to2 * si2;
    const int ti3 = to3 * si3;

    const int iidx1 = ti3 + ti2 + ti1 + ti0;
    const int iidx2 = iidx1 + di0_half;
    const int oidx  = to3 * so3 + to2 * so2 + to1 * so1 + to0;

    if (to0 < di0_half && to1 < di1 && to2 < di2) {
        out.ptr[oidx].x = in.ptr[iidx1];
        if (ti0 == di0_half - 1 && odd_di0)
            out.ptr[oidx].y = 0;
        else
            out.ptr[oidx].y = in.ptr[iidx2];
    } else {
        // Pad remaining elements with 0s
        out.ptr[oidx].x = 0;
        out.ptr[oidx].y = 0;
    }
}

template<typename To, typename Ti>
__global__ void padArray(Param<To> out, CParam<Ti> in) {
    const int t = blockDim.x * blockIdx.x + threadIdx.x;

    const int tMax = out.strides[3] * out.dims[3];

    if (t >= tMax) return;

    const int do1 = out.dims[1];
    const int do2 = out.dims[2];
    const int so1 = out.strides[1];
    const int so2 = out.strides[2];
    const int so3 = out.strides[3];

    const int to0 = t % so1;
    const int to1 = (t / so1) % do1;
    const int to2 = (t / so2) % do2;
    const int to3 = (t / so3);

    const int di0 = in.dims[0];
    const int di1 = in.dims[1];
    const int di2 = in.dims[2];
    const int di3 = in.dims[3];
    const int si1 = in.strides[1];
    const int si2 = in.strides[2];
    const int si3 = in.strides[3];

    const int ti0 = to0;
    const int ti1 = to1 * si1;
    const int ti2 = to2 * si2;
    const int ti3 = to3 * si3;

    const int iidx = ti3 + ti2 + ti1 + ti0;

    const int t2 = to3 * so3 + to2 * so2 + to1 * so1 + to0;

    if (to0 < di0 && to1 < di1 && to2 < di2 && to3 < di3) {
        // Copy input elements to real elements, set imaginary elements to 0
        out.ptr[t2].x = in.ptr[iidx];
        out.ptr[t2].y = 0;
    } else {
        // Pad remaining of the matrix to 0s
        out.ptr[t2].x = 0;
        out.ptr[t2].y = 0;
    }
}

template<typename convT, AF_BATCH_KIND kind>
__global__ void complexMultiply(Param<convT> out, Param<convT> in1,
                                Param<convT> in2, const int nelem) {
    const int t = blockDim.x * blockIdx.x + threadIdx.x;

    if (t >= nelem) return;

    if (kind == AF_BATCH_NONE || kind == AF_BATCH_SAME) {
        // Complex multiply each signal to equivalent filter
        const int ridx = t;

        convT c1 = in1.ptr[ridx];
        convT c2 = in2.ptr[ridx];

        out.ptr[ridx].x = c1.x * c2.x - c1.y * c2.y;
        out.ptr[ridx].y = c1.x * c2.y + c1.y * c2.x;
    } else if (kind == AF_BATCH_LHS) {
        // Complex multiply all signals to filter
        const int ridx1 = t;
        const int ridx2 = t % (in2.strides[3] * in2.dims[3]);

        convT c1 = in1.ptr[ridx1];
        convT c2 = in2.ptr[ridx2];

        out.ptr[ridx1].x = c1.x * c2.x - c1.y * c2.y;
        out.ptr[ridx1].y = c1.x * c2.y + c1.y * c2.x;
    } else if (kind == AF_BATCH_RHS) {
        // Complex multiply signal to all filters
        const int ridx1 = t % (in1.strides[3] * in1.dims[3]);
        const int ridx2 = t;

        convT c1 = in1.ptr[ridx1];
        convT c2 = in2.ptr[ridx2];

        out.ptr[ridx2].x = c1.x * c2.x - c1.y * c2.y;
        out.ptr[ridx2].y = c1.x * c2.y + c1.y * c2.x;
    }
}

template<typename To, typename Ti, bool expand, bool roundOut>
__global__ void reorderOutput(Param<To> out, Param<Ti> in, CParam<To> filter,
                              const int half_di0, const int rank,
                              const int fftScale) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    const int tMax = out.strides[3] * out.dims[3];

    if (t >= tMax) return;

    const int do1 = out.dims[1];
    const int do2 = out.dims[2];
    const int so1 = out.strides[1];
    const int so2 = out.strides[2];
    const int so3 = out.strides[3];

    const int si1 = in.strides[1];
    const int si2 = in.strides[2];
    const int si3 = in.strides[3];

    const int to0 = t % so1;
    const int to1 = (t / so1) % do1;
    const int to2 = (t / so2) % do2;
    const int to3 = (t / so3);

    int oidx = to3 * so3 + to2 * so2 + to1 * so1 + to0;

    int ti0, ti1, ti2, ti3;
    if (expand) {
        ti0 = to0;
        ti1 = to1 * si1;
        ti2 = to2 * si2;
        ti3 = to3 * si3;
    } else {
        ti0 = to0 + filter.dims[0] / 2;
        ti1 = (to1 + (rank > 1) * (filter.dims[1] / 2)) * si1;
        ti2 = (to2 + (rank > 2) * (filter.dims[2] / 2)) * si2;
        ti3 = to3 * si3;
    }

    // Divide output elements to cuFFT resulting scale, round result if output
    // type is single or double precision floating-point
    if (ti0 < half_di0) {
        // Copy top elements
        int iidx = ti3 + ti2 + ti1 + ti0;
        if (roundOut)
            out.ptr[oidx] = (To)roundf(in.ptr[iidx].x / fftScale);
        else
            out.ptr[oidx] = (To)(in.ptr[iidx].x / fftScale);
    } else if (ti0 < half_di0 + filter.dims[0] - 1) {
        // Add signal and filter elements to central part
        int iidx1 = ti3 + ti2 + ti1 + ti0;
        int iidx2 = ti3 + ti2 + ti1 + (ti0 - half_di0);
        if (roundOut)
            out.ptr[oidx] =
                (To)roundf((in.ptr[iidx1].x + in.ptr[iidx2].y) / fftScale);
        else
            out.ptr[oidx] =
                (To)((in.ptr[iidx1].x + in.ptr[iidx2].y) / fftScale);
    } else {
        // Copy bottom elements
        const int iidx = ti3 + ti2 + ti1 + (ti0 - half_di0);
        if (roundOut)
            out.ptr[oidx] = (To)roundf(in.ptr[iidx].y / fftScale);
        else
            out.ptr[oidx] = (To)(in.ptr[iidx].y / fftScale);
    }
}

}  // namespace cuda
}  // namespace arrayfire
