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
#include <math.hpp>
#include "interp.hpp"

namespace cpu
{
namespace kernel
{

template<typename InT, typename LocT, int order>
void approx1(Array<InT> output, const Array<InT> input,
             const Array<LocT> xposition, const float offGrid, af_interp_type method)
{
    InT * out = output.get();
    const LocT *xpos = xposition.get();

    const af::dim4 odims     = output.dims();
    const af::dim4 idims     = input.dims();
    const af::dim4 xdims     = xposition.dims();

    const af::dim4 ostrides  = output.strides();
    const af::dim4 istrides  = input.strides();
    const af::dim4 xstrides  = xposition.strides();

    Interp1<InT, LocT, order> interp;
    bool batch = !(xdims[1] == 1 && xdims[2] == 1 && xdims[3] == 1);

    for(dim_t idw = 0; idw < odims[3]; idw++) {
        for(dim_t idz = 0; idz < odims[2]; idz++) {
            dim_t ooffzw = idw * ostrides[3] + idz * ostrides[2];
            dim_t ioffzw = idw * istrides[3] + idz * istrides[2];
            dim_t xoffzw = idw * xstrides[3] + idz * xstrides[2];

            for(dim_t idy = 0; idy < odims[1]; idy++) {

                dim_t ooff = ooffzw + idy * ostrides[1];
                dim_t ioff = ioffzw + idy * istrides[1];
                dim_t xoff = xoffzw + idy * xstrides[1];

                for(dim_t idx = 0; idx < odims[0]; idx++) {

                    const LocT x = xpos[batch * xoff + idx];

                    if (x < 0 || idims[0] < x + 1) {
                        out[ooff + idx] = scalar<InT>(offGrid);
                    } else {
                        out[ooff + idx] = interp(input, ioff, x, method);
                    }
                }
            }
        }
    }
}

template<typename InT, typename LocT, int order>
void approx2(Array<InT> output, const Array<InT> input,
             const Array<LocT> xposition, const Array<LocT> yposition,
             float const offGrid, af_interp_type method)
{
    InT * out = output.get();
    const LocT *xpos = xposition.get();
    const LocT *ypos = yposition.get();

    af::dim4 const odims     = output.dims();
    af::dim4 const idims     = input.dims();
    af::dim4 const xdims     = xposition.dims();
    af::dim4 const ostrides  = output.strides();
    af::dim4 const istrides  = input.strides();
    af::dim4 const xstrides  = xposition.strides();
    af::dim4 const ystrides  = yposition.strides();

    Interp2<InT, LocT, order> interp;
    bool batch = !(xdims[2] == 1 && xdims[3] == 1);

    for(dim_t idw = 0; idw < odims[3]; idw++) {
        for(dim_t idz = 0; idz < odims[2]; idz++) {

            dim_t xoffzw = idw * xstrides[3] + idz * xstrides[2];
            dim_t yoffzw = idw * ystrides[3] + idz * ystrides[2];
            dim_t ooffzw = idw * ostrides[3] + idz * ostrides[2];
            dim_t ioffzw = idw * istrides[3] + idz * istrides[2];

            for(dim_t idy = 0; idy < odims[1]; idy++) {
                dim_t xoff = xoffzw * batch + idy * xstrides[1];
                dim_t yoff = yoffzw * batch + idy * ystrides[1];
                dim_t ooff = ooffzw         + idy * ostrides[1];

                for(dim_t idx = 0; idx < odims[0]; idx++) {

                    const LocT x = xpos[xoff + idx];
                    const LocT y = ypos[yoff + idx];

                    if (x < 0 || idims[0] < x + 1 ||
                        y < 0 || idims[1] < y + 1 ) {
                        out[ooff + idx] = scalar<InT>(offGrid);
                    } else {
                        out[ooff + idx] = interp(input, ioffzw, x, y, method);
                    }
                }
            }
        }
    }
}
}
}
