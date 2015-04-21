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

static const dim_type THREADS = 256;

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

    const dim_type do0 = out.dims[0];
    const dim_type do1 = out.dims[1];
    const dim_type do2 = out.dims[2];

    const dim_type do01 = do0 * do1;
    const dim_type do012 = do01 * do2;
    const dim_type do0_half = do0/2;
    const dim_type do01_half = do0_half * do1;
    const dim_type do012_half = do01_half * do2;

    const int to0 = t % do0_half;
    const int to1 = (t / do0_half) % do1;
    const int to2 = (t / do01_half) % do2;
    const int to3 = t / do012_half;

    const dim_type di0 = in.dims[0];
    const dim_type di1 = in.dims[1];
    const dim_type di2 = in.dims[2];

    const dim_type di01 = di0 * di1;
    const dim_type di012 = di01 * di2;

    const int ti0 = to0;
    const int ti1 = to1 * di0;
    const int ti2 = to2 * di01;
    const int ti3 = to3 * di012;

    const int iidx1 = ti3 + ti2 + ti1 + ti0;
    const int iidx2 = iidx1 + di0_half;
    const int oidx1 = to3*do012 + to2*do01 + to1*do0 + to0*2;
    const int oidx2 = oidx1 + 1;

    if (to0 < di0_half && to1 < di1 && to2 < di2) {
        out.ptr[oidx1] = (To)in.ptr[iidx1];
        if (ti0 == di0_half-1 && odd_di0)
            out.ptr[oidx2] = (To)0;
        else
            out.ptr[oidx2] = (To)in.ptr[iidx2];
    }
    else {
        // Pad remaining elements with 0s
        out.ptr[oidx1] = (To)0;
        out.ptr[oidx2] = (To)0;
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

    const dim_type do0 = out.dims[0]/2;
    const dim_type do1 = out.dims[1];
    const dim_type do2 = out.dims[2];

    const dim_type do01 = do0 * do1;
    const dim_type do012 = do01 * do2;

    const int to0 = t % do0;
    const int to1 = (t / do0) % do1;
    const int to2 = (t / do01) % do2;
    const int to3 = (t / do012);

    const dim_type di0 = in.dims[0];
    const dim_type di1 = in.dims[1];
    const dim_type di2 = in.dims[2];
    const dim_type di3 = in.dims[3];

    const dim_type di01 = di0 * di1;
    const dim_type di012 = di01 * di2;

    const int ti0 = to0;
    const int ti1 = to1 * di0;
    const int ti2 = to2 * di01;
    const int ti3 = to3 * di012;

    const int iidx = ti3 + ti2 + ti1 + ti0;

    const int t2 = t*2;

    if (to0 < di0 && to1 < di1 && to2 < di2 && to3 < di3) {
        // Copy input elements to real elements, set imaginary elements to 0
        out.ptr[t2]   = in.ptr[iidx];
        out.ptr[t2+1] = (To)0;
    }
    else {
        // Pad remaining of the matrix to 0s
        out.ptr[t2]   = (To)0;
        out.ptr[t2+1] = (To)0;
    }
}

template<typename T, ConvolveBatchKind kind>
__global__ void complexMultiply(
    Param<T> out,
    Param<T> in1,
    Param<T> in2,
    const dim_type nelem)
{
    const int t = blockDim.x * blockIdx.x + threadIdx.x;

    if (t >= nelem)
        return;

    if (kind == ONE2ONE || kind == MANY2MANY) {
        // Complex multiply each signal to equivalent filter
        const int ridx = t * 2;
        const int iidx = t * 2 + 1;

        T a = in1.ptr[ridx];
        T b = in1.ptr[iidx];
        T c = in2.ptr[ridx];
        T d = in2.ptr[iidx];

        T ac = a*c;
        T bd = b*d;

        out.ptr[ridx] = ac - bd;
        out.ptr[iidx] = (a+b) * (c+d) - ac - bd;
    }
    else if (kind == MANY2ONE) {
        // Complex multiply all signals to filter
        const int ridx1 = t * 2;
        const int iidx1 = t * 2 + 1;
        const int ridx2 = (t*2)   % (in2.strides[3] * in2.dims[3]);
        const int iidx2 = (t*2+1) % (in2.strides[3] * in2.dims[3]);

        T a = in1.ptr[ridx1];
        T b = in1.ptr[iidx1];
        T c = in2.ptr[ridx2];
        T d = in2.ptr[iidx2];

        T ac = a*c;
        T bd = b*d;

        out.ptr[ridx1] = ac - bd;
        out.ptr[iidx1] = (a+b) * (c+d) - ac - bd;
    }
    else if (kind == ONE2MANY) {
        // Complex multiply signal to all filters
        const int ridx1 = (t*2)   % (in1.strides[3] * in1.dims[3]);
        const int iidx1 = (t*2+1) % (in1.strides[3] * in1.dims[3]);
        const int ridx2 = t * 2;
        const int iidx2 = t * 2 + 1;

        T a = in1.ptr[ridx1];
        T b = in1.ptr[iidx1];
        T c = in2.ptr[ridx2];
        T d = in2.ptr[iidx2];

        T ac = a*c;
        T bd = b*d;

        out.ptr[ridx2] = ac - bd;
        out.ptr[iidx2] = (a+b) * (c+d) - ac - bd;
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

    const dim_type do0 = out.dims[0];
    const dim_type do1 = out.dims[1];
    const dim_type do2 = out.dims[2];

    const dim_type do01 = do0 * do1;
    const dim_type do012 = do01 * do2;

    const dim_type di0 = in.dims[0];
    const dim_type di1 = in.dims[1];
    const dim_type di2 = in.dims[2];

    const dim_type di01 = di0 * di1;
    const dim_type di012 = di01 * di2;

    const int to0 = t % do0;
    const int to1 = (t / do0) % do1;
    const int to2 = (t / do01) % do2;
    const int to3 = (t / do012);

    int oidx = to3*do012 + to2*do01 + to1*do0 + to0;

    int ti0, ti1, ti2, ti3;
    if (expand) {
        ti0 = to0;
        ti1 = to1 * di0;
        ti2 = to2 * di01;
        ti3 = to3 * di012;
    }
    else {
        ti0 = to0 + filter.dims[0]/2;
        ti1 = (to1 + (baseDim > 1)*(filter.dims[1]/2)) * di0;
        ti2 = (to2 + (baseDim > 2)*(filter.dims[2]/2)) * di01;
        ti3 = to3 * di012;
    }

    // Divide output elements to cuFFT resulting scale, round result if output
    // type is single or double precision floating-point
    if (ti0 < half_di0) {
        // Copy top elements
        int iidx = ti3 + ti2 + ti1 + ti0 * 2;
        if (roundOut)
            out.ptr[oidx] = (To)roundf(in.ptr[iidx] / fftScale);
        else
            out.ptr[oidx] = (To)(in.ptr[iidx] / fftScale);
    }
    else if (ti0 < half_di0 + filter.dims[0] - 1) {
        // Add signal and filter elements to central part
        int iidx1 = ti3 + ti2 + ti1 + ti0 * 2;
        int iidx2 = ti3 + ti2 + ti1 + (ti0 - half_di0) * 2 + 1;
        if (roundOut)
            out.ptr[oidx] = (To)roundf((in.ptr[iidx1] + in.ptr[iidx2]) / fftScale);
        else
            out.ptr[oidx] = (To)((in.ptr[iidx1] + in.ptr[iidx2]) / fftScale);
    }
    else {
        // Copy bottom elements
        const int iidx = ti3 + ti2 + ti1 + (ti0 - half_di0) * 2 + 1;
        if (roundOut)
            out.ptr[oidx] = (To)roundf(in.ptr[iidx] / fftScale);
        else
            out.ptr[oidx] = (To)(in.ptr[iidx] / fftScale);
    }
}

template<typename T, typename convT, bool isDouble, bool roundOut,
         dim_type baseDim, bool expand>
void fftconvolve(Param<T> out,
                 CParam<T> sig,
                 CParam<T> filter,
                 ConvolveBatchKind kind)
{
    dim_type *sd = sig.dims;
    dim_type *fd = filter.dims;
    dim_type fftScale = 1;

    Param<convT> packed;
    dim_type fft_dims[baseDim];

    // Pack both signal and filter on same memory array, this will ensure
    // better use of batched cuFFT capabilities
    for (dim_type k = 0; k < 4; k++) {
        if (k < baseDim)
            packed.dims[k] = nextpow2((unsigned)(sd[k] + fd[k] - 1));
        else if (k == baseDim)
            packed.dims[k] = sd[k] + fd[k];
        else
            packed.dims[k] = 1;

        packed.strides[k] = (k == 0) ? 1 : packed.strides[k - 1] * packed.dims[k - 1];

        if (k < baseDim) {
            // Invert dimensions order, as cuFFT expects it this way
            fft_dims[baseDim-k-1] = (k == 0) ? packed.dims[k] / 2 : packed.dims[k];
            fftScale *= fft_dims[baseDim-k-1];
        }
    }

    dim_type packed_elem = packed.strides[3] * packed.dims[3];

    // Create cuFFT plan
    cufftHandle plan;
    cufftResult res;
    if (isDouble)
        res = cufftPlanMany(&plan, baseDim, fft_dims, NULL, 0, 0,
                            NULL, 0, 0, CUFFT_Z2Z, packed.dims[baseDim]);
    else
        res = cufftPlanMany(&plan, baseDim, fft_dims, NULL, 0, 0,
                            NULL, 0, 0, CUFFT_C2C, packed.dims[baseDim]);

    // If there was no memory available, call garbage collector and try again
    if (res != CUFFT_SUCCESS) {
        garbageCollect();
        if (isDouble)
            CUFFT_CHECK(cufftPlanMany(&plan, baseDim, fft_dims, NULL, 0, 0,
                                      NULL, 0, 0, CUFFT_Z2Z, packed.dims[baseDim]));
        else
            CUFFT_CHECK(cufftPlanMany(&plan, baseDim, fft_dims, NULL, 0, 0,
                                      NULL, 0, 0, CUFFT_C2C, packed.dims[baseDim]));
    }

    packed.ptr = memAlloc<convT>(packed_elem);

    Param<convT> sig_tmp, filter_tmp;
    sig_tmp.dims[0] = filter_tmp.dims[0] = packed.dims[0];
    sig_tmp.strides[0] = filter_tmp.strides[0] = 1;

    for (dim_type k = 1; k < 4; k++) {
        if (k < baseDim) {
            sig_tmp.dims[k]    = packed.dims[k];
            filter_tmp.dims[k] = packed.dims[k];
        }
        else {
            sig_tmp.dims[k]    = sd[k];
            filter_tmp.dims[k] = fd[k];
        }

        sig_tmp.strides[k]    = sig_tmp.strides[k - 1] * sig_tmp.dims[k - 1];
        filter_tmp.strides[k] = filter_tmp.strides[k - 1] * filter_tmp.dims[k - 1];
    }

    // Calculate memory offsets for packed signal and filter
    sig_tmp.ptr = packed.ptr;
    filter_tmp.ptr = packed.ptr + sig_tmp.strides[3] * sig_tmp.dims[3];

    dim_type sig_packed_elem = sig_tmp.strides[3] * sig_tmp.dims[3];
    dim_type filter_packed_elem = filter_tmp.strides[3] * filter_tmp.dims[3];

    // Number of packed complex elements in dimension 0
    dim_type sig_half_d0 = divup(sd[0], 2);
    bool sig_half_d0_odd = (sd[0] % 2 == 1);

    dim3 threads(THREADS);
    dim3 blocks(divup(sig_packed_elem / 2, threads.x));

    // Pack signal in a complex matrix where first dimension is half the input
    // (allows faster FFT computation) and pad array to a power of 2 with 0s
    packData<convT, T><<<blocks, threads>>>(sig_tmp, sig, sig_half_d0, sig_half_d0_odd);
    POST_LAUNCH_CHECK();

    blocks = dim3(divup(filter_packed_elem / 2, threads.x));

    // Pad filter array with 0s
    padArray<convT, T><<<blocks, threads>>>(filter_tmp, filter);
    POST_LAUNCH_CHECK();

    // Compute forward FFT
    if (isDouble)
        CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)packed.ptr,
                                 (cufftDoubleComplex*)packed.ptr, CUFFT_FORWARD));
    else
        CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex*)packed.ptr,
                                 (cufftComplex*)packed.ptr, CUFFT_FORWARD));

    dim_type mul_elem = (sig_packed_elem < filter_packed_elem) ?
                        filter_packed_elem / 2 : sig_packed_elem / 2;
    blocks = dim3(divup(mul_elem, threads.x));

    // Multiply filter and signal FFT arrays
    switch(kind) {
        case ONE2ONE:
            complexMultiply<convT, ONE2ONE  ><<<blocks, threads>>>
                (sig_tmp, sig_tmp, filter_tmp, mul_elem);
            break;
        case MANY2ONE:
            complexMultiply<convT, MANY2ONE ><<<blocks, threads>>>
                (sig_tmp, sig_tmp, filter_tmp, mul_elem);
            break;
        case ONE2MANY:
            complexMultiply<convT, ONE2MANY ><<<blocks, threads>>>
                (filter_tmp, sig_tmp, filter_tmp, mul_elem);
            break;
        case MANY2MANY:
            complexMultiply<convT, MANY2MANY><<<blocks, threads>>>
                (sig_tmp, sig_tmp, filter_tmp, mul_elem);
            break;
    }
    POST_LAUNCH_CHECK();

    // Compute inverse FFT
    if (isDouble)
        CUFFT_CHECK(cufftExecZ2Z(plan, (cufftDoubleComplex*)packed.ptr,
                                 (cufftDoubleComplex*)packed.ptr, CUFFT_INVERSE));
    else
        CUFFT_CHECK(cufftExecC2C(plan, (cufftComplex*)packed.ptr,
                                 (cufftComplex*)packed.ptr, CUFFT_INVERSE));

    CUFFT_CHECK(cufftDestroy(plan));

    blocks = dim3(divup(out.strides[3] * out.dims[3], threads.x));
    if (kind == ONE2MANY) {
        reorderOutput<T, convT, expand, roundOut><<<blocks, threads>>>
            (out, filter_tmp, filter, sig_half_d0, baseDim, fftScale);
    }
    else {
        reorderOutput<T, convT, expand, roundOut><<<blocks, threads>>>
            (out, sig_tmp, filter, sig_half_d0, baseDim, fftScale);
    }
    POST_LAUNCH_CHECK();

    memFree(packed.ptr);
}

} // namespace kernel

} // namespace cuda
