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
#include <common/Binary.hpp>
#include <utility.hpp>
#include <limits>

namespace arrayfire {
namespace cpu {
namespace kernel {
template<typename T>
void getOffsets(std::vector<dim_t>& offsets, const af::dim4& strides,
                const CParam<T>& mask) {
    const af::dim4 fstrides = mask.strides();
    const T* filter         = mask.get();
    const dim_t dim0 = mask.dims()[0], dim1 = mask.dims()[1];
    const dim_t R0 = dim0 / 2;
    const dim_t R1 = dim1 / 2;

    offsets.reserve(mask.dims().elements());
    for (dim_t j = 0; j < dim1; ++j) {
        for (dim_t i = 0; i < dim0; ++i) {
            if (filter[getIdx(fstrides, i, j)] > (T)0) {
                dim_t offset = (j - R1) * strides[1] + (i - R0) * strides[0];
                offsets.push_back(offset);
            }
        }
    }
}

template<typename T, bool IsDilation>
struct MorphFilterOp {
    T operator()(const T& a, const T& b) {
        return IsDilation ? std::max(a, b) : std::min(a, b);
    }
};

template<typename T, bool IsDilation>
void morph(Param<T> paddedOut, CParam<T> paddedIn, CParam<T> mask) {
    MorphFilterOp<T, IsDilation> filterOp;
    T init = IsDilation ? common::Binary<T, af_max_t>::init()
                        : common::Binary<T, af_min_t>::init();

    const af::dim4 ostrides = paddedOut.strides();
    T* outData              = paddedOut.get();
    const af::dim4 istrides = paddedIn.strides();
    const af::dim4 dims     = paddedIn.dims();
    const T* inData         = paddedIn.get();

    std::vector<dim_t> offsets;
    getOffsets(offsets, istrides, mask);

    const dim_t batchSize = dims[0] * dims[1];
    const int batchCount  = dims[2] * dims[3];
    for (int b = 0; b < batchCount; ++b) {
        for (dim_t n = 0; n < batchSize; ++n) {
            T filterResult = init;
            for (size_t oi = 0; oi < offsets.size(); ++oi) {
                dim_t x = n + offsets[oi];
                if (x >= 0 && x < batchSize)
                    filterResult = filterOp(filterResult, inData[x]);
            }
            outData[n] = filterResult;
        }
        outData += ostrides[2];
        inData += istrides[2];
    }
}

template<typename T, bool IsDilation>
void morph3d(Param<T> out, CParam<T> in, CParam<T> mask) {
    const af::dim4 dims     = in.dims();
    const af::dim4 window   = mask.dims();
    const dim_t R0          = window[0] / 2;
    const dim_t R1          = window[1] / 2;
    const dim_t R2          = window[2] / 2;
    const af::dim4 istrides = in.strides();
    const af::dim4 fstrides = mask.strides();
    const dim_t bCount      = dims[3];
    const af::dim4 ostrides = out.strides();
    T* outData              = out.get();
    const T* inData         = in.get();
    const T* filter         = mask.get();

    T init = IsDilation ? common::Binary<T, af_max_t>::init()
                        : common::Binary<T, af_min_t>::init();

    for (dim_t batchId = 0; batchId < bCount; ++batchId) {
        // either channels or batch is handled by outer most loop
        for (dim_t k = 0; k < dims[2]; ++k) {
            // k steps along 3rd dimension
            for (dim_t j = 0; j < dims[1]; ++j) {
                // j steps along 2nd dimension
                for (dim_t i = 0; i < dims[0]; ++i) {
                    // i steps along 1st dimension
                    T filterResult = init;

                    // wk, wj,wi steps along 2nd & 1st dimensions of filter
                    // window respectively
                    for (dim_t wk = 0; wk < window[2]; wk++) {
                        for (dim_t wj = 0; wj < window[1]; wj++) {
                            for (dim_t wi = 0; wi < window[0]; wi++) {
                                dim_t offk = k + wk - R2;
                                dim_t offj = j + wj - R1;
                                dim_t offi = i + wi - R0;

                                T maskValue =
                                    filter[getIdx(fstrides, wi, wj, wk)];

                                if ((maskValue > (T)0) && offi >= 0 &&
                                    offj >= 0 && offk >= 0 && offi < dims[0] &&
                                    offj < dims[1] && offk < dims[2]) {
                                    T inValue = inData[getIdx(istrides, offi,
                                                              offj, offk)];

                                    if (IsDilation)
                                        filterResult =
                                            std::max(filterResult, inValue);
                                    else
                                        filterResult =
                                            std::min(filterResult, inValue);
                                }

                            }  // window 1st dimension loop ends here
                        }      // window 1st dimension loop ends here
                    }          // filter window loop ends here

                    outData[getIdx(ostrides, i, j, k)] = filterResult;
                }  // 1st dimension loop ends here
            }      // 2nd dimension loop ends here
        }          // 3rd dimension loop ends here
        // next iteration will be next batch if any
        outData += ostrides[3];
        inData += istrides[3];
    }
}
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
