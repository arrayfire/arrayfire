/*******************************************************
 * Copyright (c) 2015,  ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <blas.hpp>
#include <handle.hpp>
#include <cast.hpp>
#include <Array.hpp>
#include <math.hpp>

namespace cpu
{
namespace kernel
{


template<typename InT, typename LocT, af_interp_type Method>
struct approx2_op
{
    void operator()(InT *out, af::dim4 const & odims, dim_t const oElems,
              InT const * const in,  af::dim4 const & idims, dim_t const iElems,
              LocT const * const pos, af::dim4 const & pdims, LocT const * const qos, af::dim4 const & qdims,
              af::dim4 const & ostrides, af::dim4 const & istrides,
              af::dim4 const & pstrides, af::dim4 const & qstrides,
              float const offGrid, bool const pBatch,
              dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        return;
    }
};

template<typename InT, typename LocT>
struct approx2_op<InT, LocT, AF_INTERP_NEAREST>
{
    void operator()(InT *out, af::dim4 const & odims, dim_t const oElems,
              InT const * const in,  af::dim4 const & idims, dim_t const iElems,
              LocT const * const pos, af::dim4 const & pdims, LocT const * const qos, af::dim4 const & qdims,
              af::dim4 const & ostrides, af::dim4 const & istrides,
              af::dim4 const & pstrides, af::dim4 const & qstrides,
              float const offGrid, bool const pBatch,
              dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        dim_t pmId = idy * pstrides[1] + idx;
        dim_t qmId = idy * qstrides[1] + idx;
        if(pBatch) {
            pmId += idw * pstrides[3] + idz * pstrides[2];
            qmId += idw * qstrides[3] + idz * qstrides[2];
        }

        bool gFlag = false;
        LocT const x = pos[pmId], y = qos[qmId];
        if (x < 0 || y < 0 || idims[0] < x+1 || idims[1] < y+1) {
            gFlag = true;
        }

        dim_t const omId = idw * ostrides[3] + idz * ostrides[2]
                         + idy * ostrides[1] + idx;
        if(gFlag) {
            out[omId] = scalar<InT>(offGrid);
        } else {
            dim_t const grid_x = round(x), grid_y = round(y); // nearest grid
            dim_t const imId = idw * istrides[3] + idz * istrides[2] +
                            grid_y * istrides[1] + grid_x;
            out[omId] = in[imId];
        }
    }
};

template<typename InT, typename LocT>
struct approx2_op<InT, LocT, AF_INTERP_LINEAR>
{
    void operator()(InT *out, af::dim4 const & odims, dim_t const oElems,
              InT const * const in,  af::dim4 const & idims, dim_t const iElems,
              LocT const * const pos, af::dim4 const & pdims, LocT const * const qos, af::dim4 const & qdims,
              af::dim4 const & ostrides, af::dim4 const & istrides,
              af::dim4 const & pstrides, af::dim4 const & qstrides,
              float const offGrid, bool const pBatch,
              dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        dim_t pmId = idy * pstrides[1] + idx;
        dim_t qmId = idy * qstrides[1] + idx;
        if(pBatch) {
            pmId += idw * pstrides[3] + idz * pstrides[2];
            qmId += idw * qstrides[3] + idz * qstrides[2];
        }

        bool gFlag = false;
        LocT const x = pos[pmId], y = qos[qmId];
        if (x < 0 || y < 0 || idims[0] < x+1 || idims[1] < y+1) {
            gFlag = true;
        }

        dim_t const grid_x = floor(x),   grid_y = floor(y);   // nearest grid
        LocT const off_x  = x - grid_x, off_y  = y - grid_y; // fractional offset

        // Check if pVal and pVal + 1 are both valid indices
        bool condY = (y < idims[1] - 1);
        bool condX = (x < idims[0] - 1);

        // Compute wieghts used
        LocT wt00 = ((LocT)1.0 - off_x) * ((LocT)1.0 - off_y);
        LocT wt10 = (condY) ? ((LocT)1.0 - off_x) * (off_y) : 0;
        LocT wt01 = (condX) ? (off_x) * ((LocT)1.0 - off_y) : 0;
        LocT wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

        LocT wt = wt00 + wt10 + wt01 + wt11;
        InT zero = scalar<InT>(0);

        dim_t const omId = idw * ostrides[3] + idz * ostrides[2]
                         + idy * ostrides[1] + idx;
        if(gFlag) {
            out[omId] = scalar<InT>(offGrid);
        } else {
            dim_t ioff = idw * istrides[3] + idz * istrides[2]
                    + grid_y * istrides[1] + grid_x;

            // Compute Weighted Values
            InT y00 =                    wt00 * in[ioff];
            InT y10 = (condY) ?          wt10 * in[ioff + istrides[1]]     : zero;
            InT y01 = (condX) ?          wt01 * in[ioff + 1]               : zero;
            InT y11 = (condX && condY) ? wt11 * in[ioff + istrides[1] + 1] : zero;

            InT yo = y00 + y10 + y01 + y11;

            // Write Final Value
            out[omId] = (yo / wt);
        }
    }
};

template<typename InT, typename LocT> inline static
InT cubicInterpolate(InT p[4], LocT x) {
    return p[1] + scalar<InT>(0.5) * x * (p[2] - p[0] + x * (scalar<InT>(2.0) * p[0] - scalar<InT>(5.0) * p[1] + scalar<InT>(4.0) * p[2] - p[3] + x*(scalar<InT>(3.0)*(p[1] - p[2]) + p[3] - p[0])));
}

template<typename InT, typename LocT> inline static
InT bicubicInterpolate(InT p[4][4], LocT x, LocT y) {
    InT arr[4];
    arr[0] = cubicInterpolate(p[0], x);
    arr[1] = cubicInterpolate(p[1], x);
    arr[2] = cubicInterpolate(p[2], x);
    arr[3] = cubicInterpolate(p[3], x);
    return cubicInterpolate(arr, y);
}

template<typename InT, typename LocT>
struct approx2_op<InT, LocT, AF_INTERP_CUBIC>
{
    void operator()(InT *out, af::dim4 const & odims, dim_t const oElems,
              InT const * const in,  af::dim4 const & idims, dim_t const iElems,
              LocT const * const pos, af::dim4 const & pdims, LocT const * const qos, af::dim4 const & qdims,
              af::dim4 const & ostrides, af::dim4 const & istrides,
              af::dim4 const & pstrides, af::dim4 const & qstrides,
              float const offGrid, bool const pBatch,
              dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        dim_t pmId = idy * pstrides[1] + idx;
        dim_t qmId = idy * qstrides[1] + idx;
        if(pBatch) {
            pmId += idw * pstrides[3] + idz * pstrides[2];
            qmId += idw * qstrides[3] + idz * qstrides[2];
        }

        bool gFlag = false;
        LocT const x = pos[pmId], y = qos[qmId];
        if (x < 0 || y < 0 || idims[0] < x+1 || idims[1] < y+1) {
            gFlag = true;
        }

        dim_t const grid_x = floor(x),   grid_y = floor(y);   // nearest grid
        LocT const off_x  = x - grid_x, off_y  = y - grid_y; // fractional offset

        // Check if pVal - 1, pVal, and pVal + 1, pVal + 2 are both valid indices
        bool condY  = (y > 0 && y < idims[1] - 3);
        bool condX  = (x > 0 && x < idims[0] - 3);
        bool condXl = (x > 0);
        bool condYl = (y > 0);
        bool condXrr= (x > idims[0] - 3);
        bool condYrr = (y > idims[1] - 3);

        // Compute wieghts used
        dim_t const omId = idw * ostrides[3] + idz * ostrides[2]
                         + idy * ostrides[1] + idx;
        if(gFlag) {
            out[omId] = scalar<InT>(offGrid);
        } else {
            dim_t ioff = idw * istrides[3] + idz * istrides[2]
                    + grid_y * istrides[1] + grid_x;

            InT patch[4][4];

            //inner square
            //assumption is that inner patch consisting of 4 points is minimum requirement for bicubic interpolation
            patch[1][1] = in[ioff];
            patch[1][2] = in[ioff + 1];
            patch[2][1] = in[ioff + istrides[1]];
            patch[2][2] = in[ioff + istrides[1] + 1];

            //top left corner 
            patch[0][0] = (condXl && condYl)? in[ioff - istrides[1] - 1] : scalar<InT>(offGrid);
            //
            //leftside
            patch[1][0] = (condXl)? in[ioff - 1] : scalar<InT>(offGrid);
            patch[2][0] = (condXl)? in[ioff + istrides[1] -1] : scalar<InT>(offGrid);
            //bottom left corner
            patch[3][0] = (condXl && !condYrr)? in[ioff + 2*istrides[1] - 1] : scalar<InT>(offGrid);
            //
            //top side
            patch[0][1] = (condYl)?  in[ioff-istrides[1]] : scalar<InT>(offGrid);
            patch[0][2] = (condYl)?  in[ioff-istrides[1]+1] : scalar<InT>(offGrid);
            //top right corner
            patch[0][3] = (condYl && !condXrr)? in[ioff - istrides[1] + 1] : scalar<InT>(offGrid);
            //
            //right side
            patch[1][3] = (condXrr)? scalar<InT>(offGrid) : in[ioff + 2];
            patch[2][3] = (condXrr)? scalar<InT>(offGrid) : in[ioff + istrides[1] + 2];
            //bottom right corner
            //
            patch[3][3] = (!condXrr && !condYrr)? in[ioff + 2 * istrides[1] + 2] : scalar<InT>(offGrid);
            //bottom corner
            patch[3][1] = (condYrr)? scalar<InT>(offGrid) : in[ioff + 2*istrides[1]];
            patch[3][2] = (condYrr)? scalar<InT>(offGrid) : in[ioff + 2*istrides[1] + 1];


            // Write Final Value
            out[omId] = bicubicInterpolate(patch, off_x, off_y);
        }
    }
};

template<typename InT, typename LocT, af_interp_type Method>
void approx2(Array<InT> output, Array<InT> const input,
             Array<LocT> const position, Array<LocT> const qosition,
             float const offGrid)
{
    InT * out = output.get();
    InT const * const in  = input.get();
    LocT const * const pos = position.get();
    LocT const * const qos = qosition.get();
    af::dim4 const odims     = output.dims();
    af::dim4 const idims     = input.dims();
    af::dim4 const pdims     = position.dims();
    af::dim4 const qdims     = qosition.dims();
    af::dim4 const ostrides  = output.strides();
    af::dim4 const istrides  = input.strides();
    af::dim4 const pstrides  = position.strides();
    af::dim4 const qstrides  = qosition.strides();
    dim_t const oElems   = output.elements();
    dim_t const iElems   = input.elements();

    approx2_op<InT, LocT, Method> op;
    bool pBatch = !(pdims[2] == 1 && pdims[3] == 1);

    for(dim_t w = 0; w < odims[3]; w++) {
        for(dim_t z = 0; z < odims[2]; z++) {
            for(dim_t y = 0; y < odims[1]; y++) {
                for(dim_t x = 0; x < odims[0]; x++) {
                    op(out, odims, oElems, in, idims, iElems, pos, pdims, qos, qdims,
                       ostrides, istrides, pstrides, qstrides, offGrid, pBatch, x, y, z, w);
                }
            }
        }
    }
}

}
}
