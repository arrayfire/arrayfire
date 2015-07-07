/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <backend.hpp>
#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <err_cufft.hpp>
#include <debug_cuda.hpp>
#include <Param.hpp>
#include <memory.hpp>
#include <cufft.h>

namespace cuda
{

namespace kernel
{

static const int THREADS = 256;

template<typename To, typename Ti>
__global__ void packData(
    Param<To> out,
    CParam<Ti> in,
    const int di0_half,
    const bool odd_di0)
{
    const int t = blockDim.x * blockIdx.x + threadIdx.x;

    const int tMax = out.strides[3] * out.dims[3];

    if (t >= tMax)
        return;

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
    const int oidx = to3*so3 + to2*so2 + to1*so1 + to0;

    if (to0 < di0_half && to1 < di1 && to2 < di2) {
        out.ptr[oidx].x = in.ptr[iidx1];
        if (ti0 == di0_half-1 && odd_di0)
            out.ptr[oidx].y = 0;
        else
            out.ptr[oidx].y = in.ptr[iidx2];
    }
    else {
        // Pad remaining elements with 0s
        out.ptr[oidx].x = 0;
        out.ptr[oidx].y = 0;
    }
}

template<typename To, typename Ti>
__global__ void padArray(
    Param<To> out,
    CParam<Ti> in)
{
    const int t = blockDim.x * blockIdx.x + threadIdx.x;

    const int tMax = out.strides[3] * out.dims[3];

    if (t >= tMax)
        return;

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

    const int t2 = to3*so3 + to2*so2 + to1*so1 + to0;

    if (to0 < di0 && to1 < di1 && to2 < di2 && to3 < di3) {
        // Copy input elements to real elements, set imaginary elements to 0
        out.ptr[t2].x = in.ptr[iidx];
        out.ptr[t2].y = 0;
    }
    else {
        // Pad remaining of the matrix to 0s
        out.ptr[t2].x = 0;
        out.ptr[t2].y = 0;
    }
}

template<typename convT, ConvolveBatchKind kind>
__global__ void complexMultiply(
    Param<convT> out,
    Param<convT> in1,
    Param<convT> in2,
    const int nelem)
{
    const int t = blockDim.x * blockIdx.x + threadIdx.x;

    if (t >= nelem)
        return;

    if (kind == CONVOLVE_BATCH_NONE || kind == CONVOLVE_BATCH_SAME) {
        // Complex multiply each signal to equivalent filter
        const int ridx = t;

        convT c1 = in1.ptr[ridx];
        convT c2 = in2.ptr[ridx];

        out.ptr[ridx].x = c1.x*c2.x - c1.y*c2.y;
        out.ptr[ridx].y = (c1.x+c1.y) * (c2.x+c2.y) - c1.x*c2.x - c1.y*c2.y;
    }
    else if (kind == CONVOLVE_BATCH_SIGNAL) {
        // Complex multiply all signals to filter
        const int ridx1 = t;
        const int ridx2 = t % (in2.strides[3] * in2.dims[3]);

        convT c1 = in1.ptr[ridx1];
        convT c2 = in2.ptr[ridx2];

        out.ptr[ridx1].x = c1.x*c2.x - c1.y*c2.y;
        out.ptr[ridx1].y = (c1.x+c1.y) * (c2.x+c2.y) - c1.x*c2.x - c1.y*c2.y;
    }
    else if (kind == CONVOLVE_BATCH_KERNEL) {
        // Complex multiply signal to all filters
        const int ridx1 = t % (in1.strides[3] * in1.dims[3]);
        const int ridx2 = t;

        convT c1 = in1.ptr[ridx1];
        convT c2 = in2.ptr[ridx2];

        out.ptr[ridx2].x = c1.x*c2.x - c1.y*c2.y;
        out.ptr[ridx2].y = (c1.x+c1.y) * (c2.x+c2.y) - c1.x*c2.x - c1.y*c2.y;
    }
}

template<typename To, typename Ti, bool expand, bool roundOut>
__global__ void reorderOutput(
    Param<To> out,
    Param<Ti> in,
    CParam<To> filter,
    const int half_di0,
    const int baseDim,
    const int fftScale)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;

    const int tMax = out.strides[3] * out.dims[3];

    if (t >= tMax)
        return;

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

    int oidx = to3*so3 + to2*so2 + to1*so1 + to0;

    int ti0, ti1, ti2, ti3;
    if (expand) {
        ti0 = to0;
        ti1 = to1 * si1;
        ti2 = to2 * si2;
        ti3 = to3 * si3;
    }
    else {
        ti0 = to0 + filter.dims[0]/2;
        ti1 = (to1 + (baseDim > 1)*(filter.dims[1]/2)) * si1;
        ti2 = (to2 + (baseDim > 2)*(filter.dims[2]/2)) * si2;
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
    }
    else if (ti0 < half_di0 + filter.dims[0] - 1) {
        // Add signal and filter elements to central part
        int iidx1 = ti3 + ti2 + ti1 + ti0;
        int iidx2 = ti3 + ti2 + ti1 + (ti0 - half_di0);
        if (roundOut)
            out.ptr[oidx] = (To)roundf((in.ptr[iidx1].x + in.ptr[iidx2].y) / fftScale);
        else
            out.ptr[oidx] = (To)((in.ptr[iidx1].x + in.ptr[iidx2].y) / fftScale);
    }
    else {
        // Copy bottom elements
        const int iidx = ti3 + ti2 + ti1 + (ti0 - half_di0);
        if (roundOut)
            out.ptr[oidx] = (To)roundf(in.ptr[iidx].y / fftScale);
        else
            out.ptr[oidx] = (To)(in.ptr[iidx].y / fftScale);
    }
}

template<typename convT, typename T>
void packDataHelper(Param<convT> sig_packed,
                    Param<convT> filter_packed,
                    CParam<T> sig,
                    CParam<T> filter,
                    const int baseDim)
{
    dim_t *sd = sig.dims;

    int sig_packed_elem = sig_packed.strides[3] * sig_packed.dims[3];
    int filter_packed_elem = filter_packed.strides[3] * filter_packed.dims[3];

    // Number of packed complex elements in dimension 0
    int sig_half_d0 = divup(sd[0], 2);
    bool sig_half_d0_odd = (sd[0] % 2 == 1);

    dim3 threads(THREADS);
    dim3 blocks(divup(sig_packed_elem, threads.x));

    // Pack signal in a complex matrix where first dimension is half the input
    // (allows faster FFT computation) and pad array to a power of 2 with 0s
    packData<convT, T><<<blocks, threads>>>(sig_packed, sig, sig_half_d0, sig_half_d0_odd);
    POST_LAUNCH_CHECK();

    blocks = dim3(divup(filter_packed_elem, threads.x));

    // Pad filter array with 0s
    padArray<convT, T><<<blocks, threads>>>(filter_packed, filter);
    POST_LAUNCH_CHECK();
}

template<typename T, typename convT>
void complexMultiplyHelper(Param<T> out,
                           Param<convT> sig_packed,
                           Param<convT> filter_packed,
                           CParam<T> sig,
                           CParam<T> filter,
                           ConvolveBatchKind kind)
{
    int sig_packed_elem = sig_packed.strides[3] * sig_packed.dims[3];
    int filter_packed_elem = filter_packed.strides[3] * filter_packed.dims[3];

    dim3 threads(THREADS);
    dim3 blocks(divup(sig_packed_elem / 2, threads.x));

    int mul_elem = (sig_packed_elem < filter_packed_elem) ?
                        filter_packed_elem : sig_packed_elem;
    blocks = dim3(divup(mul_elem, threads.x));

    // Multiply filter and signal FFT arrays
    switch(kind) {
        case CONVOLVE_BATCH_NONE:
            complexMultiply<convT, CONVOLVE_BATCH_NONE  ><<<blocks, threads>>>
                (sig_packed, sig_packed, filter_packed, mul_elem);
            break;
        case CONVOLVE_BATCH_SIGNAL:
            complexMultiply<convT, CONVOLVE_BATCH_SIGNAL ><<<blocks, threads>>>
                (sig_packed, sig_packed, filter_packed, mul_elem);
            break;
        case CONVOLVE_BATCH_KERNEL:
            complexMultiply<convT, CONVOLVE_BATCH_KERNEL ><<<blocks, threads>>>
                (filter_packed, sig_packed, filter_packed, mul_elem);
            break;
        case CONVOLVE_BATCH_SAME:
            complexMultiply<convT, CONVOLVE_BATCH_SAME><<<blocks, threads>>>
                (sig_packed, sig_packed, filter_packed, mul_elem);
            break;
        case CONVOLVE_BATCH_UNSUPPORTED:
        default:
            break;
    }
    POST_LAUNCH_CHECK();
}

template<typename T, typename convT, bool roundOut, int baseDim, bool expand>
void reorderOutputHelper(Param<T> out,
                         Param<convT> packed,
                         CParam<T> sig,
                         CParam<T> filter,
                         ConvolveBatchKind kind)
{
    dim_t *sd = sig.dims;
    int fftScale = 1;

    // Calculate the scale by which to divide cuFFT results
    for (int k = 0; k < baseDim; k++)
        fftScale *= packed.dims[k];

    // Number of packed complex elements in dimension 0
    int sig_half_d0 = divup(sd[0], 2);

    dim3 threads(THREADS);
    dim3 blocks(divup(out.strides[3] * out.dims[3], threads.x));

    reorderOutput<T, convT, expand, roundOut><<<blocks, threads>>>
        (out, packed, filter, sig_half_d0, baseDim, fftScale);
    POST_LAUNCH_CHECK();
}

} // namespace kernel

} // namespace cuda
