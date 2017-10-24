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
#include "interp.hpp"

namespace cpu
{
namespace kernel
{

template<typename InT, typename LocT, int order>
void approx1(Param<InT> yo, CParam<InT> yi,
             CParam<LocT> xo, const int xdim,
             const LocT &xi_beg, const LocT &xi_step,
             const float offGrid, af_interp_type method)
{
    InT *yo_ptr = yo.get();
    const LocT *xo_ptr = xo.get();

    const af::dim4 yo_dims     = yo.dims();
    const af::dim4 yi_dims     = yi.dims();
    const af::dim4 xo_dims     = xo.dims();

    const af::dim4 yo_strides  = yo.strides();
    const af::dim4 yi_strides  = yi.strides();
    const af::dim4 xo_strides  = xo.strides();

    Interp1<InT, LocT, order> interp;
    bool batch[] = {xo_dims[0] > 1, xo_dims[1] > 1, xo_dims[2] > 1, xo_dims[3] > 1};

    for(dim_t idw = 0; idw < yo_dims[3]; idw++) {
        for(dim_t idz = 0; idz < yo_dims[2]; idz++) {
            dim_t ooffzw = idw * yo_strides[3] + idz * yo_strides[2];
            dim_t ioffzw = idw * yi_strides[3] + idz * yi_strides[2];
            dim_t xoffzw = idw * xo_strides[3] * batch[3] + idz * xo_strides[2] * batch[2];

            for(dim_t idy = 0; idy < yo_dims[1]; idy++) {

                dim_t ooff = ooffzw + idy * yo_strides[1];
                dim_t ioff = ioffzw + idy * yi_strides[1];
                dim_t xoff = xoffzw + idy * xo_strides[1] * batch[1];

                for(dim_t idx = 0; idx < yo_dims[0]; idx++) {

                    const LocT x = (xo_ptr[xoff + idx * batch[0]] - xi_beg) / xi_step;

                    // FIXME: Only cubic interpolation is doing clamping
                    // We need to make it consistent across all methods
                    // Not changing the behavior because tests will fail
                    bool clamp = order == 3;

                    if (x < 0 || yi_dims[xdim] < x + 1) {
                        yo_ptr[ooff + idx] = scalar<InT>(offGrid);
                    } else {
                        interp(yo, ooff + idx, yi, ioff, x, method, 1, clamp);
                    }
                }
            }
        }
    }
}

template<typename InT, typename LocT, int order>
void approx2(Param<InT> zo, CParam<InT> zi,
             CParam<LocT> xo, const int xdim,
             CParam<LocT> yo, const int ydim,
             const LocT &xi_beg, const LocT &xi_step,
             const LocT &yi_beg, const LocT &yi_step,
             float const offGrid, af_interp_type method)
{
    InT *zo_ptr = zo.get();
    const LocT *xo_ptr = xo.get();
    const LocT *yo_ptr = yo.get();

    af::dim4 const zo_dims     = zo.dims();
    af::dim4 const zi_dims     = zi.dims();
    af::dim4 const xo_dims     = xo.dims();
    af::dim4 const zo_strides  = zo.strides();
    af::dim4 const zi_strides  = zi.strides();
    af::dim4 const xo_strides  = xo.strides();
    af::dim4 const yo_strides  = yo.strides();

    Interp2<InT, LocT, order> interp;
    bool batch[] = {xo_dims[0] > 1, xo_dims[1] > 1, xo_dims[2] > 1, xo_dims[3] > 1};

    for(dim_t idw = 0; idw < zo_dims[3]; idw++) {
        for(dim_t idz = 0; idz < zo_dims[2]; idz++) {

            dim_t xoffzw = idw * xo_strides[3] * batch[3] + idz * xo_strides[2] * batch[2];
            dim_t yoffzw = idw * yo_strides[3] * batch[3] + idz * yo_strides[2] * batch[2];
            dim_t ooffzw = idw * zo_strides[3] + idz * zo_strides[2];
            dim_t ioffzw = idw * zi_strides[3] + idz * zi_strides[2];

            for(dim_t idy = 0; idy < zo_dims[1]; idy++) {
                dim_t xoff = xoffzw + idy * xo_strides[1] * batch[1];
                dim_t yoff = yoffzw + idy * yo_strides[1] * batch[1];
                dim_t ooff = ooffzw + idy * zo_strides[1];

                for(dim_t idx = 0; idx < zo_dims[0]; idx++) {

                    const LocT x = (xo_ptr[xoff + idx] - xi_beg) / xi_step;
                    const LocT y = (yo_ptr[yoff + idx] - yi_beg) / yi_step;

                    // FIXME: Only cubic interpolation is doing clamping
                    // We need to make it consistent across all methods
                    // Not changing the behavior because tests will fail
                    bool clamp = order == 3;

                    if (x < 0 || zi_dims[0] < x + 1 ||
                        y < 0 || zi_dims[1] < y + 1 ) {
                        zo_ptr[ooff + idx] = scalar<InT>(offGrid);
                    } else {
                        interp(zo, ooff + idx, zi, ioffzw, x, y, method, 1, clamp);
                    }
                }
            }
        }
    }
}
}
}
