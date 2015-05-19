/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
#include <dispatch.hpp>
#include <fft.hpp>
#include <err_cpu.hpp>
#include <fftw3.h>
#include <copy.hpp>
#include <convolve_common.hpp>

namespace cpu
{

template<typename To, typename Ti>
void packData(To* out_ptr, const af::dim4& od, const af::dim4& os,
              Array<Ti> const& in)
{
    const af::dim4 id = in.dims();
    const af::dim4 is = in.strides();
    const Ti* in_ptr = in.get();

    int id0_half = divup(id[0], 2);
    bool odd_id0 = (id[0] % 2 == 1);

    for (int d3 = 0; d3 < (int)od[3]; d3++) {
        for (int d2 = 0; d2 < (int)od[2]; d2++) {
            for (int d1 = 0; d1 < (int)od[1]; d1++) {
                for (int d0 = 0; d0 < (int)od[0] / 2; d0++) {
                    const dim_t oidx = d3*os[3] + d2*os[2] + d1*os[1] + d0*2;

                    if (d0 < (int)id0_half && d1 < (int)id[1] && d2 < (int)id[2] && d3 < (int)id[3]) {
                        const dim_t iidx = d3*is[3] + d2*is[2] + d1*is[1] + d0;
                        out_ptr[oidx]   = (To)in_ptr[iidx];
                        if (d0 == id0_half-1 && odd_id0)
                            out_ptr[oidx+1] = (To)0;
                        else
                            out_ptr[oidx+1] = (To)in_ptr[iidx+id0_half];
                    }
                    else {
                        // Pad remaining elements with 0s
                        out_ptr[oidx]   = (To)0;
                        out_ptr[oidx+1] = (To)0;
                    }
                }
            }
        }
    }
}

template<typename To, typename Ti>
void padArray(To* out_ptr, const af::dim4& od, const af::dim4& os,
              Array<Ti> const& in)
{
    const af::dim4 id = in.dims();
    const af::dim4 is = in.strides();
    const Ti* in_ptr = in.get();

    for (int d3 = 0; d3 < (int)od[3]; d3++) {
        for (int d2 = 0; d2 < (int)od[2]; d2++) {
            for (int d1 = 0; d1 < (int)od[1]; d1++) {
                for (int d0 = 0; d0 < (int)od[0] / 2; d0++) {
                    const dim_t oidx = d3*os[3] + d2*os[2] + d1*os[1] + d0*2;

                    if (d0 < (int)id[0] && d1 < (int)id[1] && d2 < (int)id[2] && d3 < (int)id[3]) {
                        // Copy input elements to real elements, set imaginary elements to 0
                        const dim_t iidx = d3*is[3] + d2*is[2] + d1*is[1] + d0;
                        out_ptr[oidx]   = (To)in_ptr[iidx];
                        out_ptr[oidx+1] = (To)0;
                    }
                    else {
                        // Pad remaining of the matrix to 0s
                        out_ptr[oidx]   = (To)0;
                        out_ptr[oidx+1] = (To)0;
                    }
                }
            }
        }
    }
}

template<typename T>
void complexMultiply(T* out_ptr, const af::dim4& od, const af::dim4& os,
                     T* in1_ptr, const af::dim4& i1d, const af::dim4& i1s,
                     T* in2_ptr, const af::dim4& i2d, const af::dim4& i2s,
                     ConvolveBatchKind kind)
{
    for (int d3 = 0; d3 < (int)od[3]; d3++) {
        for (int d2 = 0; d2 < (int)od[2]; d2++) {
            for (int d1 = 0; d1 < (int)od[1]; d1++) {
                for (int d0 = 0; d0 < (int)od[0] / 2; d0++) {
                    if (kind == ONE2ONE || kind == MANY2MANY) {
                        // Complex multiply each signal to equivalent filter
                        const int ridx = d3*os[3] + d2*os[2] + d1*os[1] + d0*2;
                        const int iidx = ridx + 1;

                        T a = in1_ptr[ridx];
                        T b = in1_ptr[iidx];
                        T c = in2_ptr[ridx];
                        T d = in2_ptr[iidx];

                        T ac = a*c;
                        T bd = b*d;

                        out_ptr[ridx] = ac - bd;
                        out_ptr[iidx] = (a+b) * (c+d) - ac - bd;
                    }
                    else if (kind == MANY2ONE) {
                        // Complex multiply all signals to filter
                        const int ridx1 = d3*os[3] + d2*os[2] + d1*os[1] + d0*2;
                        const int iidx1 = ridx1 + 1;
                        const int ridx2 = ridx1 % (i2s[3] * i2d[3]);
                        const int iidx2 = iidx1 % (i2s[3] * i2d[3]);

                        T a = in1_ptr[ridx1];
                        T b = in1_ptr[iidx1];
                        T c = in2_ptr[ridx2];
                        T d = in2_ptr[iidx2];

                        T ac = a*c;
                        T bd = b*d;

                        out_ptr[ridx1] = ac - bd;
                        out_ptr[iidx1] = (a+b) * (c+d) - ac - bd;
                    }
                    else if (kind == ONE2MANY) {
                        // Complex multiply signal to all filters
                        const int ridx2 = d3*os[3] + d2*os[2] + d1*os[1] + d0*2;
                        const int iidx2 = ridx2 + 1;
                        const int ridx1 = ridx2 % (i1s[3] * i1d[3]);
                        const int iidx1 = iidx2 % (i1s[3] * i1d[3]);

                        T a = in1_ptr[ridx1];
                        T b = in1_ptr[iidx1];
                        T c = in2_ptr[ridx2];
                        T d = in2_ptr[iidx2];

                        T ac = a*c;
                        T bd = b*d;

                        out_ptr[ridx2] = ac - bd;
                        out_ptr[iidx2] = (a+b) * (c+d) - ac - bd;
                    }
                }
            }
        }
    }
}

template<typename To, typename Ti, bool roundOut>
void reorderOutput(To* out_ptr, const af::dim4& od, const af::dim4& os,
                   const Ti* in_ptr, const af::dim4& id, const af::dim4& is,
                   const af::dim4& fd, const int half_di0, const int baseDim,
                   const int fftScale, const bool expand)
{
    for (int d3 = 0; d3 < (int)od[3]; d3++) {
        for (int d2 = 0; d2 < (int)od[2]; d2++) {
            for (int d1 = 0; d1 < (int)od[1]; d1++) {
                for (int d0 = 0; d0 < (int)od[0]; d0++) {
                    int id0, id1, id2, id3;
                    if (expand) {
                        id0 = d0;
                        id1 = d1 * is[1];
                        id2 = d2 * is[2];
                        id3 = d3 * is[3];
                    }
                    else {
                        id0 = d0 + fd[0]/2;
                        id1 = (d1 + (baseDim > 1)*(fd[1]/2)) * is[1];
                        id2 = (d2 + (baseDim > 2)*(fd[2]/2)) * is[2];
                        id3 = d3 * is[3];
                    }

                    int oidx = d3*os[3] + d2*os[2] + d1*os[1] + d0;

                    // Divide output elements to cuFFT resulting scale, round result if output
                    // type is single or double precision floating-point
                    if (id0 < half_di0) {
                        // Copy top elements
                        int iidx = id3 + id2 + id1 + id0 * 2;
                        if (roundOut)
                            out_ptr[oidx] = (To)roundf((float)(in_ptr[iidx] / fftScale));
                        else
                            out_ptr[oidx] = (To)(in_ptr[iidx] / fftScale);
                    }
                    else if (id0 < half_di0 + (int)fd[0] - 1) {
                        // Add signal and filter elements to central part
                        int iidx1 = id3 + id2 + id1 + id0 * 2;
                        int iidx2 = id3 + id2 + id1 + (id0 - half_di0) * 2 + 1;
                        if (roundOut)
                            out_ptr[oidx] = (To)roundf((float)((in_ptr[iidx1] + in_ptr[iidx2]) / fftScale));
                        else
                            out_ptr[oidx] = (To)((in_ptr[iidx1] + in_ptr[iidx2]) / fftScale);
                    }
                    else {
                        // Copy bottom elements
                        const int iidx = id3 + id2 + id1 + (id0 - half_di0) * 2 + 1;
                        if (roundOut)
                            out_ptr[oidx] = (To)roundf((float)(in_ptr[iidx] / fftScale));
                        else
                            out_ptr[oidx] = (To)(in_ptr[iidx] / fftScale);
                    }
                }
            }
        }
    }
}

template<typename T, typename convT, typename cT, bool isDouble, bool roundOut, dim_t baseDim>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter,
                     const bool expand, ConvolveBatchKind kind)
{
    const af::dim4 sd = signal.dims();
    const af::dim4 fd = filter.dims();

    dim_t fftScale = 1;

    af::dim4 packed_dims;
    int fft_dims[baseDim];
    af::dim4 sig_tmp_dims, sig_tmp_strides;
    af::dim4 filter_tmp_dims, filter_tmp_strides;

    // Pack both signal and filter on same memory array, this will ensure
    // better use of batched cuFFT capabilities
    for (dim_t k = 0; k < 4; k++) {
        if (k < baseDim)
            packed_dims[k] = nextpow2((unsigned)(sd[k] + fd[k] - 1));
        else if (k == baseDim)
            packed_dims[k] = sd[k] + fd[k];
        else
            packed_dims[k] = 1;

        if (k < baseDim) {
            fft_dims[baseDim-k-1] = (k == 0) ? packed_dims[k] / 2 : packed_dims[k];
            fftScale *= fft_dims[baseDim-k-1];
        }
    }

    Array<convT> packed = createEmptyArray<convT>(packed_dims);
    convT *packed_ptr = packed.get();

    const af::dim4 packed_strides = packed.strides();

    sig_tmp_dims[0]    = filter_tmp_dims[0] = packed_dims[0];
    sig_tmp_strides[0] = filter_tmp_strides[0] = 1;

    for (dim_t k = 1; k < 4; k++) {
        if (k < baseDim) {
            sig_tmp_dims[k]    = packed_dims[k];
            filter_tmp_dims[k] = packed_dims[k];
        }
        else {
            sig_tmp_dims[k]    = sd[k];
            filter_tmp_dims[k] = fd[k];
        }

        sig_tmp_strides[k]    = sig_tmp_strides[k - 1] * sig_tmp_dims[k - 1];
        filter_tmp_strides[k] = filter_tmp_strides[k - 1] * filter_tmp_dims[k - 1];
    }

    // Calculate memory offsets for packed signal and filter
    convT *sig_tmp_ptr    = packed_ptr;
    convT *filter_tmp_ptr = packed_ptr + sig_tmp_strides[3] * sig_tmp_dims[3];

    // Number of packed complex elements in dimension 0
    dim_t sig_half_d0 = divup(sd[0], 2);

    // Pack signal in a complex matrix where first dimension is half the input
    // (allows faster FFT computation) and pad array to a power of 2 with 0s
    packData<convT, T>(sig_tmp_ptr, sig_tmp_dims, sig_tmp_strides, signal);

    // Pad filter array with 0s
    padArray<convT, T>(filter_tmp_ptr, filter_tmp_dims, filter_tmp_strides, filter);

    // Compute forward FFT
    if (isDouble) {
        fftw_plan plan = fftw_plan_many_dft(baseDim,
                                            fft_dims,
                                            packed_dims[baseDim],
                                            (fftw_complex*)packed.get(),
                                            NULL,
                                            packed_strides[0],
                                            packed_strides[baseDim] / 2,
                                            (fftw_complex*)packed.get(),
                                            NULL,
                                            packed_strides[0],
                                            packed_strides[baseDim] / 2,
                                            FFTW_FORWARD,
                                            FFTW_ESTIMATE);

        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }
    else {
        fftwf_plan plan = fftwf_plan_many_dft(baseDim,
                                              fft_dims,
                                              packed_dims[baseDim],
                                              (fftwf_complex*)packed.get(),
                                              NULL,
                                              packed_strides[0],
                                              packed_strides[baseDim] / 2,
                                              (fftwf_complex*)packed.get(),
                                              NULL,
                                              packed_strides[0],
                                              packed_strides[baseDim] / 2,
                                              FFTW_FORWARD,
                                              FFTW_ESTIMATE);

        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }

    // Multiply filter and signal FFT arrays
    if (kind == ONE2MANY)
        complexMultiply<convT>(filter_tmp_ptr, filter_tmp_dims, filter_tmp_strides,
                               sig_tmp_ptr, sig_tmp_dims, sig_tmp_strides,
                               filter_tmp_ptr, filter_tmp_dims, filter_tmp_strides,
                               kind);
    else
        complexMultiply<convT>(sig_tmp_ptr, sig_tmp_dims, sig_tmp_strides,
                               sig_tmp_ptr, sig_tmp_dims, sig_tmp_strides,
                               filter_tmp_ptr, filter_tmp_dims, filter_tmp_strides,
                               kind);

    // Compute inverse FFT
    if (isDouble) {
        fftw_plan plan = fftw_plan_many_dft(baseDim,
                                            fft_dims,
                                            packed_dims[baseDim],
                                            (fftw_complex*)packed.get(),
                                            NULL,
                                            packed_strides[0],
                                            packed_strides[baseDim] / 2,
                                            (fftw_complex*)packed.get(),
                                            NULL,
                                            packed_strides[0],
                                            packed_strides[baseDim] / 2,
                                            FFTW_BACKWARD,
                                            FFTW_ESTIMATE);

        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }
    else {
        fftwf_plan plan = fftwf_plan_many_dft(baseDim,
                                              fft_dims,
                                              packed_dims[baseDim],
                                              (fftwf_complex*)packed.get(),
                                              NULL,
                                              packed_strides[0],
                                              packed_strides[baseDim] / 2,
                                              (fftwf_complex*)packed.get(),
                                              NULL,
                                              packed_strides[0],
                                              packed_strides[baseDim] / 2,
                                              FFTW_BACKWARD,
                                              FFTW_ESTIMATE);

        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    }

    // Compute output dimensions
    dim4 oDims(1);
    if (expand) {
        for(dim_t d=0; d<4; ++d) {
            if (kind==ONE2ONE || kind==ONE2MANY) {
                oDims[d] = sd[d]+fd[d]-1;
            } else {
                oDims[d] = (d<baseDim ? sd[d]+fd[d]-1 : sd[d]);
            }
        }
    } else {
        oDims = sd;
        if (kind==ONE2MANY) {
            for (dim_t i=baseDim; i<4; ++i)
                oDims[i] = fd[i];
        }
    }

    Array<T> out = createEmptyArray<T>(oDims);
    T* out_ptr = out.get();
    const af::dim4 out_dims = out.dims();
    const af::dim4 out_strides = out.strides();

    const af::dim4 filter_dims = filter.dims();

    // Reorder the output
    if (kind == ONE2MANY) {
        reorderOutput<T, convT, roundOut>
            (out_ptr, out_dims, out_strides,
             filter_tmp_ptr, filter_tmp_dims, filter_tmp_strides,
             filter_dims, sig_half_d0, baseDim, fftScale, expand);
    }
    else {
        reorderOutput<T, convT, roundOut>
            (out_ptr, out_dims, out_strides,
             sig_tmp_ptr, sig_tmp_dims, sig_tmp_strides,
             filter_dims, sig_half_d0, baseDim, fftScale, expand);
    }

    return out;
}

#define INSTANTIATE(T, convT, cT, isDouble, roundOut)                                                   \
    template Array<T> fftconvolve <T, convT, cT, isDouble, roundOut, 1>                                 \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, ConvolveBatchKind kind);    \
    template Array<T> fftconvolve <T, convT, cT, isDouble, roundOut, 2>                                 \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, ConvolveBatchKind kind);    \
    template Array<T> fftconvolve <T, convT, cT, isDouble, roundOut, 3>                                 \
        (Array<T> const& signal, Array<T> const& filter, const bool expand, ConvolveBatchKind kind);

INSTANTIATE(double, double, cdouble, true , false)
INSTANTIATE(float , float,  cfloat,  false, false)
INSTANTIATE(uint  , float,  cfloat,  false, true)
INSTANTIATE(int   , float,  cfloat,  false, true)
INSTANTIATE(uchar , float,  cfloat,  false, true)
INSTANTIATE(char  , float,  cfloat,  false, true)

} // namespace cpu
