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

#include <algorithm>
#include <vector>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T, af::borderType Pad>
void medfilt1(Param<T> out, CParam<T> in, dim_t w_wid) {
    constexpr bool IsValidPadType = (Pad == AF_PAD_ZERO || Pad == AF_PAD_SYM);

    const af::dim4 dims     = in.dims();
    const af::dim4 istrides = in.strides();
    const af::dim4 ostrides = out.strides();

    std::vector<T> wind_vals;
    wind_vals.reserve(w_wid);

    for (int b3 = 0; b3 < (int)dims[3]; b3++) {
        T const* in_ptr = in.get() + b3 * istrides[3];
        T* out_ptr      = out.get() + b3 * ostrides[3];

        for (int b2 = 0; b2 < (int)dims[2]; b2++) {
            for (int col = 0; col < (int)dims[1]; col++) {
                int ocol_off = col * ostrides[1];

                for (int row = 0; row < (int)dims[0]; row++) {
                    wind_vals.clear();
                    for (int wi = 0; wi < (int)w_wid; ++wi) {
                        int im_row = row + wi - w_wid / 2;
                        int im_roff;
                        switch (Pad) {
                            case AF_PAD_ZERO:
                                im_roff = im_row * istrides[0];
                                if (im_row < 0 || im_row >= (int)dims[0])
                                    wind_vals.push_back(0);
                                else
                                    wind_vals.push_back(in_ptr[im_roff]);
                                break;
                            case AF_PAD_SYM: {
                                if (im_row < 0) { im_row *= -1; }

                                if (im_row >= (int)dims[0]) {
                                    im_row = 2 * ((int)dims[0] - 1) - im_row;
                                }

                                im_roff = im_row * istrides[0];
                                wind_vals.push_back(in_ptr[im_roff]);
                            } break;
                            default:
                                static_assert(IsValidPadType,
                                              "Unsupported padding type");
                                break;
                        }
                    }

                    int off = wind_vals.size() / 2;
                    std::stable_sort(wind_vals.begin(), wind_vals.end());
                    if (wind_vals.size() % 2 == 0)
                        out_ptr[ocol_off + row * ostrides[0]] =
                            (wind_vals[off] + wind_vals[off - 1]) / 2;
                    else {
                        out_ptr[ocol_off + row * ostrides[0]] = wind_vals[off];
                    }
                }
            }
            in_ptr += istrides[2];
            out_ptr += ostrides[2];
        }
    }
}

template<typename T, af::borderType Pad>
void medfilt2(Param<T> out, CParam<T> in, dim_t w_len, dim_t w_wid) {
    constexpr bool IsValidPadType = (Pad == AF_PAD_ZERO || Pad == AF_PAD_SYM);

    const af::dim4 dims     = in.dims();
    const af::dim4 istrides = in.strides();
    const af::dim4 ostrides = out.strides();

    std::vector<T> wind_vals;
    wind_vals.reserve(w_len * w_wid);

    for (int b3 = 0; b3 < (int)dims[3]; b3++) {
        T const* in_ptr = in.get() + b3 * istrides[3];
        T* out_ptr      = out.get() + b3 * ostrides[3];

        for (int b2 = 0; b2 < (int)dims[2]; b2++) {
            for (int col = 0; col < (int)dims[1]; col++) {
                int ocol_off = col * ostrides[1];

                for (int row = 0; row < (int)dims[0]; row++) {
                    wind_vals.clear();

                    for (int wj = 0; wj < (int)w_wid; ++wj) {
                        bool isColOff = false;

                        int im_col  = col + wj - w_wid / 2;
                        int im_coff = 0;
                        switch (Pad) {
                            case AF_PAD_ZERO:
                                im_coff = im_col * istrides[1];
                                if (im_col < 0 || im_col >= (int)dims[1])
                                    isColOff = true;
                                break;
                            case AF_PAD_SYM: {
                                if (im_col < 0) {
                                    im_col *= -1;
                                    isColOff = true;
                                }

                                if (im_col >= (int)dims[1]) {
                                    im_col   = 2 * ((int)dims[1] - 1) - im_col;
                                    isColOff = true;
                                }

                                im_coff = im_col * istrides[1];
                            } break;
                            default:
                                static_assert(IsValidPadType,
                                              "Unsupported padding type");
                                break;
                        }

                        for (int wi = 0; wi < (int)w_len; ++wi) {
                            bool isRowOff = false;

                            int im_row  = row + wi - w_len / 2;
                            int im_roff = 0;
                            switch (Pad) {
                                case AF_PAD_ZERO:
                                    im_roff = im_row * istrides[0];
                                    if (im_row < 0 || im_row >= (int)dims[0])
                                        isRowOff = true;
                                    break;
                                case AF_PAD_SYM: {
                                    if (im_row < 0) {
                                        im_row *= -1;
                                        isRowOff = true;
                                    }

                                    if (im_row >= (int)dims[0]) {
                                        im_row =
                                            2 * ((int)dims[0] - 1) - im_row;
                                        isRowOff = true;
                                    }

                                    im_roff = im_row * istrides[0];
                                } break;
                                default:
                                    static_assert(IsValidPadType,
                                                  "Unsupported padding type");
                                    break;
                            }

                            if (isRowOff || isColOff) {
                                switch (Pad) {
                                    case AF_PAD_ZERO:
                                        wind_vals.push_back(0);
                                        break;
                                    case AF_PAD_SYM:
                                        wind_vals.push_back(
                                            in_ptr[im_coff + im_roff]);
                                        break;
                                    default:
                                        static_assert(
                                            IsValidPadType,
                                            "Unsupported padding type");
                                        break;
                                }
                            } else
                                wind_vals.push_back(in_ptr[im_coff + im_roff]);
                        }
                    }

                    std::stable_sort(wind_vals.begin(), wind_vals.end());
                    int off = wind_vals.size() / 2;
                    if (wind_vals.size() % 2 == 0)
                        out_ptr[ocol_off + row * ostrides[0]] =
                            (wind_vals[off] + wind_vals[off - 1]) / 2;
                    else
                        out_ptr[ocol_off + row * ostrides[0]] = wind_vals[off];
                }
            }
            in_ptr += istrides[2];
            out_ptr += ostrides[2];
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
