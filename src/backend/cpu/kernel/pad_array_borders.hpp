/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <utility.hpp>

#include <algorithm>

namespace arrayfire {
namespace cpu {
namespace kernel {
namespace {
static inline dim_t idxByndEdge(const dim_t i, const dim_t lb, const dim_t len,
                                const af::borderType btype) {
    dim_t retVal;
    switch (btype) {
        case AF_PAD_SYM: retVal = trimIndex(i - lb, len); break;
        case AF_PAD_CLAMP_TO_EDGE:
            retVal = std::max(dim_t(0), std::min(i - lb, len - 1));
            break;
        case AF_PAD_PERIODIC: {
            dim_t rem = (i - lb) % len;
            bool cond = rem < 0;
            retVal    = cond * (rem + len) + (1 - cond) * rem;
        } break;
        default: retVal = 0; break;
    }
    return retVal;
}
}  // namespace

template<typename T>
void padBorders(Param<T> out, CParam<T> in, const dim4 lBoundPadSize,
                const dim4 uBoundPadSize, const af::borderType btype) {
    const dim4& oDims = out.dims();
    const dim4& oStrs = out.strides();
    const dim4& iDims = in.dims();
    const dim4& iStrs = in.strides();

    T const* const src = in.get();
    T* dst             = out.get();

    const dim4 validRegEnds(
        oDims[0] - uBoundPadSize[0], oDims[1] - uBoundPadSize[1],
        oDims[2] - uBoundPadSize[2], oDims[3] - uBoundPadSize[3]);
    const bool isInputLinear = iStrs[0] == 1;

    /*
     * VALID REGION COPYING DOES
     * NOT NEED ANY BOUND CHECKS
     * */
    for (dim_t l = lBoundPadSize[3]; l < validRegEnds[3]; ++l) {
        dim_t oLOff = oStrs[3] * l;
        dim_t iLOff = iStrs[3] * (l - lBoundPadSize[3]);

        for (dim_t k = lBoundPadSize[2]; k < validRegEnds[2]; ++k) {
            dim_t oKOff = oStrs[2] * k;
            dim_t iKOff = iStrs[2] * (k - lBoundPadSize[2]);

            for (dim_t j = lBoundPadSize[1]; j < validRegEnds[1]; ++j) {
                dim_t oJOff = oStrs[1] * j;
                dim_t iJOff = iStrs[1] * (j - lBoundPadSize[1]);

                if (isInputLinear) {
                    T const* const sptr = src + iLOff + iKOff + iJOff;
                    T* dptr = dst + oLOff + oKOff + oJOff + lBoundPadSize[0];

                    std::copy(sptr, sptr + iDims[0], dptr);
                } else {
                    for (dim_t i = lBoundPadSize[0]; i < validRegEnds[0]; ++i) {
                        dim_t oIOff = oStrs[0] * i;
                        dim_t iIOff = iStrs[0] * (i - lBoundPadSize[0]);

                        dst[oLOff + oKOff + oJOff + oIOff] =
                            src[iLOff + iKOff + iJOff + iIOff];
                    }
                }
            }  // second dimension loop
        }      // third dimension loop
    }          // fourth dimension loop

    // If we have to do zero padding,
    // just return as the output is filled with
    // zeros during allocation
    if (btype == AF_PAD_ZERO) return;

    /*
     * PADDED REGIONS NEED BOUND
     * CHECKS; FOLLOWING NESTED
     * LOOPS SHALL ONLY PROCESS
     * PADDED REGIONS AND SKIP REST
     * */
    for (dim_t l = 0; l < oDims[3]; ++l) {
        bool skipL  = (l >= lBoundPadSize[3] && l < validRegEnds[3]);
        dim_t oLOff = oStrs[3] * l;
        dim_t iLOff =
            iStrs[3] * idxByndEdge(l, lBoundPadSize[3], iDims[3], btype);
        for (dim_t k = 0; k < oDims[2]; ++k) {
            bool skipK  = (k >= lBoundPadSize[2] && k < validRegEnds[2]);
            dim_t oKOff = oStrs[2] * k;
            dim_t iKOff =
                iStrs[2] * idxByndEdge(k, lBoundPadSize[2], iDims[2], btype);
            for (dim_t j = 0; j < oDims[1]; ++j) {
                bool skipJ  = (j >= lBoundPadSize[1] && j < validRegEnds[1]);
                dim_t oJOff = oStrs[1] * j;
                dim_t iJOff = iStrs[1] *
                              idxByndEdge(j, lBoundPadSize[1], iDims[1], btype);
                for (dim_t i = 0; i < oDims[0]; ++i) {
                    bool skipI = (i >= lBoundPadSize[0] && i < validRegEnds[0]);
                    if (skipI && skipJ && skipK && skipL) continue;

                    dim_t oIOff = oStrs[0] * i;
                    dim_t iIOff = iStrs[0] * idxByndEdge(i, lBoundPadSize[0],
                                                         iDims[0], btype);

                    dst[oLOff + oKOff + oJOff + oIOff] =
                        src[iLOff + iKOff + iJOff + iIOff];

                }  // first dimension loop
            }      // second dimension loop
        }          // third dimension loop
    }              // fourth dimension loop
}
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
