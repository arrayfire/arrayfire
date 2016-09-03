/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <math.hpp>

namespace cpu
{
namespace kernel
{


template<typename InT, typename LocT>
InT linearInterpFunc(InT val[2], LocT frac)
{
    return (1 - frac) * val[0] + frac * val[1];
}

template<typename InT, typename LocT>
InT bilinearInterpFunc(InT val[2][2], LocT xfrac, LocT yfrac)
{
    InT res[2];
    res[0] = linearInterpFunc(val[0], xfrac);
    res[1] = linearInterpFunc(val[1], xfrac);
    return linearInterpFunc(res, yfrac);
}

template<typename InT, typename LocT>
InT cubicInterpFunc(InT val[4], LocT xfrac)
{
    InT a0 =
        scalar<InT>(-0.5) * val[0] + scalar<InT>( 1.5) * val[1] +
        scalar<InT>(-1.5) * val[2] + scalar<InT>( 0.5) * val[3];

    InT a1 =
        scalar<InT>( 1.0) * val[0] + scalar<InT>(-2.5) * val[1] +
        scalar<InT>( 2.0) * val[2] + scalar<InT>(-0.5) * val[3];

    InT a2 = scalar<InT>(-0.5) * val[0] + scalar<InT>(0.5) * val[2];

    InT a3 = val[1];

    LocT xfrac2 = xfrac * xfrac;
    LocT xfrac3 = xfrac2 * xfrac;

    return a0 * xfrac3 + a1 * xfrac2 + a2 * xfrac + a3;
}

template<typename InT, typename LocT>
InT bicubicInterpFunc(InT val[4][4], LocT xfrac, LocT yfrac) {
    InT res[4];
    res[0] = cubicInterpFunc(val[0], xfrac);
    res[1] = cubicInterpFunc(val[1], xfrac);
    res[2] = cubicInterpFunc(val[2], xfrac);
    res[3] = cubicInterpFunc(val[3], xfrac);
    return cubicInterpFunc(res, yfrac);
}

template<typename InT, typename LocT, af_interp_type>
struct Interp1
{
    InT operator()(const Array<InT> &in, int ioff, LocT x)
    {
        const InT *inptr = in.get();
        const int idx = round(x) + ioff;
        return inptr[idx];
    }
};

template<typename InT, typename LocT>
struct Interp1<InT, LocT, AF_INTERP_LINEAR>
{
    InT operator()(const Array<InT> &in, int ioff, LocT x)
    {
        const int grid_x = floor(x);    // nearest grid
        const LocT off_x = x - grid_x;    // fractional offset
        const int idx = ioff + grid_x;
        const InT *inptr = in.get();
        const dim4 idims = in.dims();
        InT val[2] = {inptr[idx], x + 1 < idims[0] ? inptr[idx + 1] : scalar<InT>(0)};
        return linearInterpFunc(val, off_x);
    }
};

template<typename InT, typename LocT>
struct Interp1<InT, LocT, AF_INTERP_CUBIC>
{
    InT operator()(const Array<InT> &in, int ioff, LocT x)
    {
        const int grid_x = floor(x);    // nearest grid
        const LocT off_x = x - grid_x;    // fractional offset
        const int idx = ioff + grid_x;
        const InT *inptr = in.get();
        const dim4 idims = in.dims();

        bool cond[4] = {grid_x - 1 >= 0, true, grid_x + 1 < idims[0], grid_x + 2 < idims[0]};
        int  off[4]  = {cond[0] ? -1 : 0, 0, cond[2] ? 1 : 0, cond[3] ? 2 : (cond[2] ? 1 : 0)};

        InT val[4];
        for (int i = 0; i < 4; i++) {
            val[i] = inptr[idx + off[i]];
        }
        return cubicInterpFunc(val, off_x);
    }
};

template<typename InT, typename LocT, af_interp_type>
struct Interp2
{
    InT operator()(const Array<InT> &in, int ioff, LocT x, LocT y)
    {
        const InT *inptr = in.get();
        const dim4 istrides = in.strides();
        const int idx = ioff + round(y) * istrides[1] + round(x);
        return inptr[idx];
    }
};

template<typename InT, typename LocT>
struct Interp2<InT, LocT, AF_INTERP_BILINEAR>
{
    InT operator()(const Array<InT> &in, int ioff, LocT x, LocT y)
    {
        const InT *inptr = in.get();
        const dim4 idims = in.dims();
        const dim4 istrides = in.strides();

        const int grid_x = floor(x);
        const LocT off_x = x - grid_x;

        const int grid_y = floor(y);
        const LocT off_y = y - grid_y;

        const int idx = ioff + grid_y * istrides[1] + grid_x;

        bool condX[2] = {true, x + 1 < idims[0]};
        bool condY[2] = {true, y + 1 < idims[1]};

        InT val[2][2];
        for (int j = 0; j < 2; j++) {
            int off_y = idx + j * istrides[1];
            for (int i = 0; i < 2; i++) {
                val[j][i] = condX[i] && condY[j] ? inptr[off_y + i] : scalar<InT>(0);
            }
        }

        return bilinearInterpFunc(val, off_x, off_y);
    }
};

template<typename InT, typename LocT>
struct Interp2<InT, LocT, AF_INTERP_LINEAR>
{
    InT operator()(const Array<InT> &in, int ioff, LocT x, LocT y)
    {
        return Interp2<InT, LocT, AF_INTERP_BILINEAR>()(in, ioff, x, y);
    }
};

template<typename InT, typename LocT>
struct Interp2<InT, LocT, AF_INTERP_BICUBIC>
{
    InT operator()(const Array<InT> &in, int ioff, LocT x, LocT y)
    {
        const InT *inptr = in.get();
        const dim4 idims = in.dims();
        const dim4 istrides = in.strides();

        const int grid_x = floor(x);
        const LocT off_x = x - grid_x;

        const int grid_y = floor(y);
        const LocT off_y = y - grid_y;

        const int idx = ioff + grid_y * istrides[1] + grid_x;

        //for bicubic interpolation, work with 4x4 val at a time
        InT val[4][4];

        // used for setting values at boundaries
        bool condX[4] = {grid_x - 1 >= 0, true, grid_x + 1 < idims[0], grid_x + 2 < idims[0]};
        bool condY[4] = {grid_y - 1 >= 0, true, grid_y + 1 < idims[1], grid_y + 2 < idims[1]};
        int  offX[4]  = {condX[0] ? -1 : 0, 0, condX[2] ? 1 : 0 , condX[3] ? 2 : (condX[2] ? 1 : 0)};
        int  offY[4]  = {condY[0] ? -1 : 0, 0, condY[2] ? 1 : 0 , condY[3] ? 2 : (condY[2] ? 1 : 0)};

        for (int j = 0; j < 4; j++) {
            int ioff_j = idx + offY[j] * istrides[1];
            for (int i = 0; i < 4; i++) {
                val[j][i] = inptr[ioff_j + offX[i]];
            }
        }

        return bicubicInterpFunc(val, off_x, off_y);
    }
};

template<typename InT, typename LocT>
struct Interp2<InT, LocT, AF_INTERP_CUBIC>
{
    InT operator()(const Array<InT> &in, int ioff, LocT x, LocT y)
    {
        return Interp2<InT, LocT, AF_INTERP_BICUBIC>()(in, ioff, x, y);
    }
};

}
}
