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

namespace cpu
{
namespace kernel
{

template<typename Ti, typename To, typename Tw>
struct MeanOp
{
    Transform<Ti, To, af_add_t> transform;
    To runningMean;
    Tw runningCount;
    MeanOp(Ti mean, Tw count) :
        transform(),
        runningMean(transform(mean)),
        runningCount(count)
    {
    }

    void operator()(Ti _newMean, Tw newCount)
    {
        To newMean = transform(_newMean);
        if ((newCount != 0) || (runningCount != 0)) {
            Tw runningScale = runningCount;
            Tw newScale = newCount;
            runningCount += newCount;
            runningScale = runningScale/runningCount;
            newScale = newScale/runningCount;
            runningMean = (runningScale*runningMean) + (newScale*newMean);
        }
    }
};

template<typename T, typename Tw, int D>
struct mean_weighted_dim
{
    void operator()(Array<T> output, const dim_t outOffset,
                    const Array< T>  input, const dim_t inOffset,
                    const Array<Tw> weight, const dim_t wtOffset, const int dim)
    {
        const af::dim4 odims    = output.dims();
        const af::dim4 ostrides = output.strides();
        const af::dim4 istrides =  input.strides();
        const af::dim4 wstrides = weight.strides();
        const int D1 = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            mean_weighted_dim<T, Tw, D1>()(output, outOffset + i * ostrides[D1],
                    input, inOffset + i * istrides[D1],
                    weight, wtOffset + i * wstrides[D1], dim);
        }
    }
};

template<typename T, typename Tw>
struct mean_weighted_dim<T, Tw, 0>
{
    void operator()(Array<T> output, const dim_t outOffset,
                    const Array< T>  input, const dim_t inOffset,
                    const Array<Tw> weight, const dim_t wtOffset, const int dim)
    {
        const af::dim4 idims = input.dims();
        const af::dim4 istrides =  input.strides();
        const af::dim4 wstrides = weight.strides();

        T const * const  in =  input.get();
        Tw const * const wt = weight.get();
        T * out = output.get();

        dim_t istride = istrides[dim];
        dim_t wstride = wstrides[dim];
        MeanOp<T, T, Tw> Op(0, 0);
        for (dim_t i = 0; i < idims[dim]; i++) {
            Op(in[inOffset + i * istride], wt[wtOffset + i * wstride]);
        }

        out[outOffset] = Op.runningMean;
    }
};

template<typename Ti, typename Tw, typename To, int D>
struct mean_dim
{
    void operator()(Array<To> output, const dim_t outOffset,
                    const Array<Ti> input, const dim_t inOffset, const int dim)
    {
        const af::dim4 odims    = output.dims();
        const af::dim4 ostrides = output.strides();
        const af::dim4 istrides =  input.strides();
        const int D1 = D - 1;
        for (dim_t i = 0; i < odims[D1]; i++) {
            mean_dim<Ti, Tw, To, D1>()(output, outOffset + i * ostrides[D1],
                                      input, inOffset + i * istrides[D1], dim);
        }
    }
};

template<typename Ti, typename Tw, typename To>
struct mean_dim<Ti, Tw, To, 0>
{
    void operator()(Array<To> output, const dim_t outOffset,
                    const Array<Ti> input, const dim_t inOffset, const int dim)
    {
        const af::dim4 idims = input.dims();
        const af::dim4 istrides =  input.strides();

        Ti const * const  in =  input.get();
        To * out = output.get();

        dim_t istride = istrides[dim];
        MeanOp<Ti, To, Tw> Op(0, 0);
        for (dim_t i = 0; i < idims[dim]; i++) {
            Op(in[inOffset + i * istride], 1);
        }

        out[outOffset] = Op.runningMean;
    }
};

}
}
