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
#include <math.hpp>
#include <af/defines.h>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename InT, typename AccT>
void one2one_1d(InT *optr, InT const *const iptr, AccT const *const fptr,
                af::dim4 const &oDims, af::dim4 const &sDims,
                af::dim4 const &fDims, af::dim4 const &sStrides,
                const bool expand) {
    dim_t start = (expand ? 0 : fDims[0] / 2);
    dim_t end   = (expand ? oDims[0] : start + sDims[0]);
    for (dim_t i = start; i < end; ++i) {
        AccT accum = 0.0;
        for (dim_t f = 0; f < fDims[0]; ++f) {
            dim_t iIdx = i - f;
            InT s_val =
                ((iIdx >= 0 && iIdx < sDims[0]) ? iptr[iIdx * sStrides[0]]
                                                : InT(0));
            accum += AccT(s_val * fptr[f]);
        }
        optr[i - start] = InT(accum);
    }
}

template<typename InT, typename AccT>
void one2one_2d(InT *optr, InT const *const iptr, AccT const *const fptr,
                af::dim4 const &oDims, af::dim4 const &sDims,
                af::dim4 const &fDims, af::dim4 const &oStrides,
                af::dim4 const &sStrides, af::dim4 const &fStrides,
                const bool expand) {
    dim_t jStart = (expand ? 0 : fDims[1] / 2);
    dim_t jEnd   = (expand ? oDims[1] : jStart + sDims[1]);
    dim_t iStart = (expand ? 0 : fDims[0] / 2);
    dim_t iEnd   = (expand ? oDims[0] : iStart + sDims[0]);

    for (dim_t j = jStart; j < jEnd; ++j) {
        dim_t joff = (j - jStart) * oStrides[1];

        for (dim_t i = iStart; i < iEnd; ++i) {
            AccT accum = AccT(0);
            for (dim_t wj = 0; wj < fDims[1]; ++wj) {
                dim_t jIdx    = j - wj;
                dim_t w_joff  = wj * fStrides[1];
                dim_t s_joff  = jIdx * sStrides[1];
                bool isJValid = (jIdx >= 0 && jIdx < sDims[1]);

                for (dim_t wi = 0; wi < fDims[0]; ++wi) {
                    dim_t iIdx = i - wi;

                    InT s_val = InT(0);
                    if (isJValid && (iIdx >= 0 && iIdx < sDims[0])) {
                        s_val = iptr[s_joff + iIdx * sStrides[0]];
                    }

                    accum += AccT(s_val * fptr[w_joff + wi * fStrides[0]]);
                }
            }
            optr[joff + i - iStart] = InT(accum);
        }
    }
}

template<typename InT, typename AccT>
void one2one_3d(InT *optr, InT const *const iptr, AccT const *const fptr,
                af::dim4 const &oDims, af::dim4 const &sDims,
                af::dim4 const &fDims, af::dim4 const &oStrides,
                af::dim4 const &sStrides, af::dim4 const &fStrides,
                const bool expand) {
    dim_t kStart = (expand ? 0 : fDims[2] / 2);
    dim_t kEnd   = (expand ? oDims[2] : kStart + sDims[2]);
    dim_t jStart = (expand ? 0 : fDims[1] / 2);
    dim_t jEnd   = (expand ? oDims[1] : jStart + sDims[1]);
    dim_t iStart = (expand ? 0 : fDims[0] / 2);
    dim_t iEnd   = (expand ? oDims[0] : iStart + sDims[0]);

    for (dim_t k = kStart; k < kEnd; ++k) {
        dim_t koff = (k - kStart) * oStrides[2];

        for (dim_t j = jStart; j < jEnd; ++j) {
            dim_t joff = (j - jStart) * oStrides[1];

            for (dim_t i = iStart; i < iEnd; ++i) {
                AccT accum = AccT(0);
                for (dim_t wk = 0; wk < fDims[2]; ++wk) {
                    dim_t kIdx    = k - wk;
                    dim_t w_koff  = wk * fStrides[2];
                    dim_t s_koff  = kIdx * sStrides[2];
                    bool isKValid = (kIdx >= 0 && kIdx < sDims[2]);

                    for (dim_t wj = 0; wj < fDims[1]; ++wj) {
                        dim_t jIdx    = j - wj;
                        dim_t w_joff  = wj * fStrides[1];
                        dim_t s_joff  = jIdx * sStrides[1];
                        bool isJValid = (jIdx >= 0 && jIdx < sDims[1]);

                        for (dim_t wi = 0; wi < fDims[0]; ++wi) {
                            dim_t iIdx = i - wi;

                            InT s_val = InT(0);
                            if (isKValid && isJValid &&
                                (iIdx >= 0 && iIdx < sDims[0])) {
                                s_val =
                                    iptr[s_koff + s_joff + iIdx * sStrides[0]];
                            }

                            accum +=
                                AccT(s_val *
                                     fptr[w_koff + w_joff + wi * fStrides[0]]);
                        }
                    }
                }
                optr[koff + joff + i - iStart] = InT(accum);
            }  // i loop ends here
        }      // j loop ends here
    }          // k loop ends here
}

template<typename InT, typename AccT>
void convolve_nd(Param<InT> out, CParam<InT> signal, CParam<AccT> filter,
                 AF_BATCH_KIND kind, const int rank, const bool expand) {
    InT *optr              = out.get();
    InT const *const iptr  = signal.get();
    AccT const *const fptr = filter.get();

    af::dim4 const oDims = out.dims();
    af::dim4 const sDims = signal.dims();
    af::dim4 const fDims = filter.dims();

    af::dim4 const oStrides = out.strides();
    af::dim4 const sStrides = signal.strides();
    af::dim4 const fStrides = filter.strides();

    dim_t out_step[AF_MAX_DIMS] = {
        0, 0, 0,
        0}; /* first value is never used, and declared for code simplicity */
    dim_t in_step[AF_MAX_DIMS] = {
        0, 0, 0,
        0}; /* first value is never used, and declared for code simplicity */
    dim_t filt_step[AF_MAX_DIMS] = {
        0, 0, 0,
        0}; /* first value is never used, and declared for code simplicity */
    dim_t batch[AF_MAX_DIMS] = {
        0, 1, 1,
        1}; /* first value is never used, and declared for code simplicity */

    for (dim_t i = 1; i < 4; ++i) {
        switch (kind) {
            case AF_BATCH_LHS:
                out_step[i] = oStrides[i];
                in_step[i]  = sStrides[i];
                if (i >= rank) batch[i] = sDims[i];
                break;
            case AF_BATCH_SAME:
                out_step[i]  = oStrides[i];
                in_step[i]   = sStrides[i];
                filt_step[i] = fStrides[i];
                if (i >= rank) batch[i] = sDims[i];
                break;
            case AF_BATCH_RHS:
                out_step[i]  = oStrides[i];
                filt_step[i] = fStrides[i];
                if (i >= rank) batch[i] = fDims[i];
                break;
            default: break;
        }
    }

    for (dim_t b3 = 0; b3 < batch[3]; ++b3) {
        for (dim_t b2 = 0; b2 < batch[2]; ++b2) {
            for (dim_t b1 = 0; b1 < batch[1]; ++b1) {
                InT *out = optr + b1 * out_step[1] + b2 * out_step[2] +
                           b3 * out_step[3];
                InT const *in =
                    iptr + b1 * in_step[1] + b2 * in_step[2] + b3 * in_step[3];
                AccT const *filt = fptr + b1 * filt_step[1] +
                                   b2 * filt_step[2] + b3 * filt_step[3];

                switch (rank) {
                    case 1:
                        one2one_1d<InT, AccT>(out, in, filt, oDims, sDims,
                                              fDims, sStrides, expand);
                        break;
                    case 2:
                        one2one_2d<InT, AccT>(out, in, filt, oDims, sDims,
                                              fDims, oStrides, sStrides,
                                              fStrides, expand);
                        break;
                    case 3:
                        one2one_3d<InT, AccT>(out, in, filt, oDims, sDims,
                                              fDims, oStrides, sStrides,
                                              fStrides, expand);
                        break;
                }
            }
        }
    }
}

template<typename InT, typename AccT, bool Expand, int ConvDim>
void convolve2_separable(InT *optr, InT const *const iptr,
                         AccT const *const fptr, af::dim4 const &oDims,
                         af::dim4 const &sDims, af::dim4 const &orgDims,
                         dim_t fDim, af::dim4 const &oStrides,
                         af::dim4 const &sStrides, dim_t fStride) {
    UNUSED(orgDims);
    UNUSED(sStrides);
    UNUSED(fStride);
    for (dim_t j = 0; j < oDims[1]; ++j) {
        dim_t jOff = j * oStrides[1];
        dim_t cj   = j + (ConvDim == 1) * (Expand ? 0 : fDim >> 1);

        for (dim_t i = 0; i < oDims[0]; ++i) {
            dim_t iOff = i * oStrides[0];
            dim_t ci   = i + (ConvDim == 0) * (Expand ? 0 : fDim >> 1);

            AccT accum = scalar<AccT>(0);

            for (dim_t f = 0; f < fDim; ++f) {
                InT f_val = fptr[f];
                InT s_val;

                if (ConvDim == 0) {
                    dim_t offi     = ci - f;
                    bool isCIValid = offi >= 0 && offi < sDims[0];
                    bool isCJValid = cj >= 0 && cj < sDims[1];
                    s_val = (isCJValid && isCIValid ? iptr[cj * sDims[0] + offi]
                                                    : scalar<InT>(0));
                } else {
                    dim_t offj     = cj - f;
                    bool isCIValid = ci >= 0 && ci < sDims[0];
                    bool isCJValid = offj >= 0 && offj < sDims[1];
                    s_val = (isCJValid && isCIValid ? iptr[offj * sDims[0] + ci]
                                                    : scalar<InT>(0));
                }

                accum += AccT(s_val * f_val);
            }
            optr[iOff + jOff] = InT(accum);
        }
    }
}

template<typename InT, typename AccT, bool Expand>
void convolve2(Param<InT> out, CParam<InT> signal, CParam<AccT> c_filter,
               CParam<AccT> r_filter, Param<InT> temp) {
    dim_t cflen = (dim_t)c_filter.dims().elements();
    dim_t rflen = (dim_t)r_filter.dims().elements();

    auto oDims = out.dims();
    auto sDims = signal.dims();

    auto oStrides = out.strides();
    auto sStrides = signal.strides();
    auto tStrides = temp.strides();

    for (dim_t b3 = 0; b3 < oDims[3]; ++b3) {
        dim_t i_b3Off = b3 * sStrides[3];
        dim_t t_b3Off = b3 * tStrides[3];
        dim_t o_b3Off = b3 * oStrides[3];

        for (dim_t b2 = 0; b2 < oDims[2]; ++b2) {
            InT const *const iptr = signal.get() + b2 * sStrides[2] + i_b3Off;
            InT *tptr             = temp.get() + b2 * tStrides[2] + t_b3Off;
            InT *optr             = out.get() + b2 * oStrides[2] + o_b3Off;

            convolve2_separable<InT, AccT, Expand, 0>(
                tptr, iptr, c_filter.get(), temp.dims(), sDims, sDims, cflen,
                tStrides, sStrides, c_filter.strides(0));

            convolve2_separable<InT, AccT, Expand, 1>(
                optr, tptr, r_filter.get(), oDims, temp.dims(), sDims, rflen,
                oStrides, tStrides, r_filter.strides(0));
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
