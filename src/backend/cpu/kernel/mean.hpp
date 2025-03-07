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
#include <common/Transform.hpp>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename Ti, typename To, typename Tw>
struct MeanOp {
    common::Transform<Ti, To, af_add_t> transform;
    To runningMean;
    Tw runningCount;
    MeanOp(Ti mean, Tw count)
        : transform(), runningMean(transform(mean)), runningCount(count) {}

    /// Prevents the optimzation of the mean calculation by some compiler flags
    /// specifically -march=native.
    [[gnu::optimize("01")]] void operator()(Ti _newMean, Tw newCount) {
        To newMean = transform(_newMean);
        if ((newCount != 0) || (runningCount != 0)) {
            Tw runningScale = runningCount;
            Tw newScale     = newCount;
            runningCount += newCount;
            runningScale = runningScale / runningCount;
            newScale     = newScale / runningCount;
            runningMean  = (runningScale * runningMean) + (newScale * newMean);
        }
    }
};

template<typename T, typename Tw, int D>
struct mean_weighted_dim {
    void operator()(Param<T> output, const dim_t outOffset,
                    const CParam<T> input, const dim_t inOffset,
                    const CParam<Tw> weight, const dim_t wtOffset,
                    const int dim) {
        const af::dim4 odims    = output.dims();
        const af::dim4 ostrides = output.strides();
        const af::dim4 istrides = input.strides();
        const af::dim4 wstrides = weight.strides();
        const int D1            = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            mean_weighted_dim<T, Tw, D1>()(output, outOffset + i * ostrides[D1],
                                           input, inOffset + i * istrides[D1],
                                           weight, wtOffset + i * wstrides[D1],
                                           dim);
        }
    }
};

template<typename T, typename Tw>
struct mean_weighted_dim<T, Tw, 0> {
    void operator()(Param<T> output, const dim_t outOffset,
                    const CParam<T> input, const dim_t inOffset,
                    const CParam<Tw> weight, const dim_t wtOffset,
                    const int dim) {
        const af::dim4 idims    = input.dims();
        const af::dim4 istrides = input.strides();
        const af::dim4 wstrides = weight.strides();

        T const* const in  = input.get();
        Tw const* const wt = weight.get();
        T* out             = output.get();

        dim_t istride = istrides[dim];
        dim_t wstride = wstrides[dim];
        MeanOp<compute_t<T>, compute_t<T>, compute_t<Tw>> Op(0, 0);
        for (dim_t i = 0; i < idims[dim]; i++) {
            Op(compute_t<T>(in[inOffset + i * istride]),
               compute_t<Tw>(wt[wtOffset + i * wstride]));
        }

        out[outOffset] = Op.runningMean;
    }
};

template<typename Ti, typename Tw, typename To, int D>
struct mean_dim {
    void operator()(Param<To> output, const dim_t outOffset,
                    const CParam<Ti> input, const dim_t inOffset,
                    const int dim) {
        const af::dim4 odims    = output.dims();
        const af::dim4 ostrides = output.strides();
        const af::dim4 istrides = input.strides();
        const int D1            = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            mean_dim<Ti, Tw, To, D1>()(output, outOffset + i * ostrides[D1],
                                       input, inOffset + i * istrides[D1], dim);
        }
    }
};

template<typename Ti, typename Tw, typename To>
struct mean_dim<Ti, Tw, To, 0> {
    void operator()(Param<To> output, const dim_t outOffset,
                    const CParam<Ti> input, const dim_t inOffset,
                    const int dim) {
        const af::dim4 idims    = input.dims();
        const af::dim4 istrides = input.strides();

        Ti const* const in = input.get();
        To* out            = output.get();

        dim_t istride = istrides[dim];
        dim_t end     = inOffset + idims[dim] * istride;
        MeanOp<compute_t<Ti>, compute_t<To>, compute_t<Tw>> Op(0, 0);
        for (dim_t i = inOffset; i < end; i += istride) {
            Op(compute_t<Ti>(in[i]), 1);
        }

        out[outOffset] = Op.runningMean;
    }
};

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
