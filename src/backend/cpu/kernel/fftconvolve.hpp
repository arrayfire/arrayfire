/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename To, typename Ti>
void packData(Param<To> out, const af::dim4 od, const af::dim4 os,
              CParam<Ti> in) {
    To* out_ptr = out.get();

    const af::dim4 id = in.dims();
    const af::dim4 is = in.strides();
    const Ti* in_ptr  = in.get();

    int id0_half = divup(id[0], 2);
    bool odd_id0 = (id[0] % 2 == 1);

    for (int d3 = 0; d3 < (int)od[3]; d3++) {
        for (int d2 = 0; d2 < (int)od[2]; d2++) {
            for (int d1 = 0; d1 < (int)od[1]; d1++) {
                for (int d0 = 0; d0 < (int)od[0] / 2; d0++) {
                    const dim_t oidx =
                        d3 * os[3] + d2 * os[2] + d1 * os[1] + d0 * 2;

                    if (d0 < (int)id0_half && d1 < (int)id[1] &&
                        d2 < (int)id[2] && d3 < (int)id[3]) {
                        const dim_t iidx =
                            d3 * is[3] + d2 * is[2] + d1 * is[1] + d0;
                        out_ptr[oidx] = (To)in_ptr[iidx];
                        if (d0 == id0_half - 1 && odd_id0)
                            out_ptr[oidx + 1] = (To)0;
                        else
                            out_ptr[oidx + 1] = (To)in_ptr[iidx + id0_half];
                    } else {
                        // Pad remaining elements with 0s
                        out_ptr[oidx]     = (To)0;
                        out_ptr[oidx + 1] = (To)0;
                    }
                }
            }
        }
    }
}

template<typename To, typename Ti>
void padArray(Param<To> out, const af::dim4 od, const af::dim4 os,
              CParam<Ti> in, const dim_t offset) {
    To* out_ptr       = out.get() + offset;
    const af::dim4 id = in.dims();
    const af::dim4 is = in.strides();
    const Ti* in_ptr  = in.get();

    for (int d3 = 0; d3 < (int)od[3]; d3++) {
        for (int d2 = 0; d2 < (int)od[2]; d2++) {
            for (int d1 = 0; d1 < (int)od[1]; d1++) {
                for (int d0 = 0; d0 < (int)od[0] / 2; d0++) {
                    const dim_t oidx =
                        d3 * os[3] + d2 * os[2] + d1 * os[1] + d0 * 2;

                    if (d0 < (int)id[0] && d1 < (int)id[1] && d2 < (int)id[2] &&
                        d3 < (int)id[3]) {
                        // Copy input elements to real elements, set imaginary
                        // elements to 0
                        const dim_t iidx =
                            d3 * is[3] + d2 * is[2] + d1 * is[1] + d0;
                        out_ptr[oidx]     = (To)in_ptr[iidx];
                        out_ptr[oidx + 1] = (To)0;
                    } else {
                        // Pad remaining of the matrix to 0s
                        out_ptr[oidx]     = (To)0;
                        out_ptr[oidx + 1] = (To)0;
                    }
                }
            }
        }
    }
}

template<typename T>
void complexMultiply(Param<T> packed, const af::dim4 sig_dims,
                     const af::dim4 sig_strides, const af::dim4 fit_dims,
                     const af::dim4 fit_strides, AF_BATCH_KIND kind,
                     const dim_t offset) {
    T* out_ptr = packed.get() + (kind == AF_BATCH_RHS ? offset : 0);
    T* in1_ptr = packed.get();
    T* in2_ptr = packed.get() + offset;

    const af::dim4& od  = (kind == AF_BATCH_RHS ? fit_dims : sig_dims);
    const af::dim4& os  = (kind == AF_BATCH_RHS ? fit_strides : sig_strides);
    const af::dim4& i1d = sig_dims;
    const af::dim4& i2d = fit_dims;
    const af::dim4& i1s = sig_strides;
    const af::dim4& i2s = fit_strides;

    for (int d3 = 0; d3 < (int)od[3]; d3++) {
        for (int d2 = 0; d2 < (int)od[2]; d2++) {
            for (int d1 = 0; d1 < (int)od[1]; d1++) {
                for (int d0 = 0; d0 < (int)od[0] / 2; d0++) {
                    if (kind == AF_BATCH_NONE || kind == AF_BATCH_SAME) {
                        // Complex multiply each signal to equivalent filter
                        const int ridx =
                            d3 * os[3] + d2 * os[2] + d1 * os[1] + d0 * 2;
                        const int iidx = ridx + 1;

                        T a = in1_ptr[ridx];
                        T b = in1_ptr[iidx];
                        T c = in2_ptr[ridx];
                        T d = in2_ptr[iidx];

                        out_ptr[ridx] = a * c - b * d;
                        out_ptr[iidx] = a * d + b * c;
                    } else if (kind == AF_BATCH_LHS) {
                        // Complex multiply all signals to filter
                        const int ridx1 =
                            d3 * os[3] + d2 * os[2] + d1 * os[1] + d0 * 2;
                        const int iidx1 = ridx1 + 1;
                        const int ridx2 = ridx1 % (i2s[3] * i2d[3]);
                        const int iidx2 = iidx1 % (i2s[3] * i2d[3]);

                        T a = in1_ptr[ridx1];
                        T b = in1_ptr[iidx1];
                        T c = in2_ptr[ridx2];
                        T d = in2_ptr[iidx2];

                        out_ptr[ridx1] = a * c - b * d;
                        out_ptr[iidx1] = a * d + b * c;
                    } else if (kind == AF_BATCH_RHS) {
                        // Complex multiply signal to all filters
                        const int ridx2 =
                            d3 * os[3] + d2 * os[2] + d1 * os[1] + d0 * 2;
                        const int iidx2 = ridx2 + 1;
                        const int ridx1 = ridx2 % (i1s[3] * i1d[3]);
                        const int iidx1 = iidx2 % (i1s[3] * i1d[3]);

                        T a = in1_ptr[ridx1];
                        T b = in1_ptr[iidx1];
                        T c = in2_ptr[ridx2];
                        T d = in2_ptr[iidx2];

                        out_ptr[ridx2] = a * c - b * d;
                        out_ptr[iidx2] = a * d + b * c;
                    }
                }
            }
        }
    }
}

template<typename To, typename Ti, int Rank, bool Expand>
void reorderHelper(To* out_ptr, const af::dim4& od, const af::dim4& os,
                   const Ti* in_ptr, const af::dim4& id, const af::dim4& is,
                   const af::dim4& fd, const int half_di0, const int fftScale) {
    constexpr bool RoundResult = std::is_integral<To>::value;

    UNUSED(id);
    for (int d3 = 0; d3 < (int)od[3]; d3++) {
        for (int d2 = 0; d2 < (int)od[2]; d2++) {
            for (int d1 = 0; d1 < (int)od[1]; d1++) {
                for (int d0 = 0; d0 < (int)od[0]; d0++) {
                    int id0, id1, id2, id3;
                    if (Expand) {
                        id0 = d0;
                        id1 = d1 * is[1];
                        id2 = d2 * is[2];
                        id3 = d3 * is[3];
                    } else {
                        id0 = d0 + fd[0] / 2;
                        id1 = (d1 + (Rank > 1) * (fd[1] / 2)) * is[1];
                        id2 = (d2 + (Rank > 2) * (fd[2] / 2)) * is[2];
                        id3 = d3 * is[3];
                    }

                    int oidx = d3 * os[3] + d2 * os[2] + d1 * os[1] + d0;

                    // Divide output elements to cuFFT resulting scale, round
                    // result if output type is single or double precision
                    // floating-point
                    if (id0 < half_di0) {
                        // Copy top elements
                        int iidx = id3 + id2 + id1 + id0 * 2;
                        if (RoundResult)
                            out_ptr[oidx] =
                                (To)roundf((float)(in_ptr[iidx] / fftScale));
                        else
                            out_ptr[oidx] = (To)(in_ptr[iidx] / fftScale);
                    } else if (id0 < half_di0 + (int)fd[0] - 1) {
                        // Add signal and filter elements to central part
                        int iidx1 = id3 + id2 + id1 + id0 * 2;
                        int iidx2 = id3 + id2 + id1 + (id0 - half_di0) * 2 + 1;
                        if (RoundResult)
                            out_ptr[oidx] = (To)roundf(
                                (float)((in_ptr[iidx1] + in_ptr[iidx2]) /
                                        fftScale));
                        else
                            out_ptr[oidx] =
                                (To)((in_ptr[iidx1] + in_ptr[iidx2]) /
                                     fftScale);
                    } else {
                        // Copy bottom elements
                        const int iidx =
                            id3 + id2 + id1 + (id0 - half_di0) * 2 + 1;
                        if (RoundResult)
                            out_ptr[oidx] =
                                (To)roundf((float)(in_ptr[iidx] / fftScale));
                        else
                            out_ptr[oidx] = (To)(in_ptr[iidx] / fftScale);
                    }
                }
            }
        }
    }
}

template<typename T, typename convT, int Rank, bool Expand>
void reorder(Param<T> out, Param<convT> packed, CParam<T> filter,
             const dim_t sig_half_d0, const dim_t fftScale,
             const dim4 sig_tmp_dims, const dim4 sig_tmp_strides,
             const dim4 filter_tmp_dims, const dim4 filter_tmp_strides,
             AF_BATCH_KIND kind) {
    T* out_ptr                 = out.get();
    const af::dim4 out_dims    = out.dims();
    const af::dim4 out_strides = out.strides();

    const af::dim4 filter_dims = filter.dims();

    convT* packed_ptr     = packed.get();
    convT* sig_tmp_ptr    = packed_ptr;
    convT* filter_tmp_ptr = packed_ptr + sig_tmp_strides[3] * sig_tmp_dims[3];

    // Reorder the output
    if (kind == AF_BATCH_RHS) {
        reorderHelper<T, convT, Rank, Expand>(
            out_ptr, out_dims, out_strides, filter_tmp_ptr, filter_tmp_dims,
            filter_tmp_strides, filter_dims, sig_half_d0, fftScale);
    } else {
        reorderHelper<T, convT, Rank, Expand>(
            out_ptr, out_dims, out_strides, sig_tmp_ptr, sig_tmp_dims,
            sig_tmp_strides, filter_dims, sig_half_d0, fftScale);
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
