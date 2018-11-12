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

namespace cpu
{
namespace kernel
{

template<int dim>
af::dim4 calcOffset(const af::dim4 dims)
{
    af::dim4 offset;
    offset[0] = (dim == 0) ? dims[0] : 0;
    offset[1] = (dim == 1) ? dims[1] : 0;
    offset[2] = (dim == 2) ? dims[2] : 0;
    offset[3] = (dim == 3) ? dims[3] : 0;
    return offset;
}

template<typename To, typename Tx, int dim>
void join_append(To *out, const Tx *X, const af::dim4 &offset,
           const af::dim4 &xdims, const af::dim4 &ost, const af::dim4 &xst)
{
    for(dim_t ow = 0; ow < xdims[3]; ow++) {
        const dim_t xW = ow * xst[3];
        const dim_t oW = (ow + offset[3]) * ost[3];

        for(dim_t oz = 0; oz < xdims[2]; oz++) {
            const dim_t xZW = xW + oz * xst[2];
            const dim_t oZW = oW + (oz + offset[2]) * ost[2];

            for(dim_t oy = 0; oy < xdims[1]; oy++) {
                const dim_t xYZW = xZW + oy * xst[1];
                const dim_t oYZW = oZW + (oy + offset[1]) * ost[1];

                for(dim_t ox = 0; ox < xdims[0]; ox++) {
                    const dim_t iMem = xYZW + ox;
                    const dim_t oMem = oYZW + (ox + offset[0]);
                    out[oMem] = X[iMem];
                }
            }
        }
    }
}

template<typename Tx, typename Ty>
void join(Param<Tx> out, const int dim, CParam<Tx> first, CParam<Ty> second)
{
    Tx* outPtr = out.get();
    const Tx* fptr = first.get();
    const Ty* sptr = second.get();

    af::dim4 zero(0,0,0,0);
    const af::dim4 fdims = first.dims();
    const af::dim4 sdims = second.dims();

    switch(dim) {
        case 0:
            join_append<Tx, Tx, 0>(outPtr, fptr, zero,
                                   fdims, out.strides(), first.strides());
            join_append<Tx, Ty, 0>(outPtr, sptr, calcOffset<0>(fdims),
                                   sdims, out.strides(), second.strides());
            break;
        case 1:
            join_append<Tx, Tx, 1>(outPtr, fptr, zero,
                                   fdims, out.strides(), first.strides());
            join_append<Tx, Ty, 1>(outPtr, sptr, calcOffset<1>(fdims),
                                   sdims, out.strides(), second.strides());
            break;
        case 2:
            join_append<Tx, Tx, 2>(outPtr, fptr, zero,
                                   fdims, out.strides(), first.strides());
            join_append<Tx, Ty, 2>(outPtr, sptr, calcOffset<2>(fdims),
                                   sdims, out.strides(), second.strides());
            break;
        case 3:
            join_append<Tx, Tx, 3>(outPtr, fptr, zero,
                                   fdims, out.strides(), first.strides());
            join_append<Tx, Ty, 3>(outPtr, sptr, calcOffset<3>(fdims),
                                   sdims, out.strides(), second.strides());
            break;
    }
}

template<typename T, int n_arrays>
void join(const int dim, Param<T> out, const std::vector<CParam<T>> inputs)
{
    af::dim4 zero(0,0,0,0);
    af::dim4 d = zero;
    switch(dim) {
        case 0:
            join_append<T, T, 0>(out.get(), inputs[0].get(), zero,
                                 inputs[0].dims(), out.strides(), inputs[0].strides());
            for(int i = 1; i < n_arrays; i++) {
                d += inputs[i - 1].dims();
                join_append<T, T, 0>(out.get(), inputs[i].get(), calcOffset<0>(d),
                                     inputs[i].dims(), out.strides(), inputs[i].strides());
            }
            break;
        case 1:
            join_append<T, T, 1>(out.get(), inputs[0].get(), zero,
                                 inputs[0].dims(), out.strides(), inputs[0].strides());
            for(int i = 1; i < n_arrays; i++) {
                d += inputs[i - 1].dims();
                join_append<T, T, 1>(out.get(), inputs[i].get(), calcOffset<1>(d),
                                     inputs[i].dims(), out.strides(), inputs[i].strides());
            }
            break;
        case 2:
            join_append<T, T, 2>(out.get(), inputs[0].get(), zero,
                                 inputs[0].dims(), out.strides(), inputs[0].strides());
            for(int i = 1; i < n_arrays; i++) {
                d += inputs[i - 1].dims();
                join_append<T, T, 2>(out.get(), inputs[i].get(), calcOffset<2>(d),
                        inputs[i].dims(), out.strides(), inputs[i].strides());
            }
            break;
        case 3:
            join_append<T, T, 3>(out.get(), inputs[0].get(), zero,
                                 inputs[0].dims(), out.strides(), inputs[0].strides());
            for(int i = 1; i < n_arrays; i++) {
                d += inputs[i - 1].dims();
                join_append<T, T, 3>(out.get(), inputs[i].get(), calcOffset<3>(d),
                                     inputs[i].dims(), out.strides(), inputs[i].strides());
            }
            break;
    }
}

}
}
