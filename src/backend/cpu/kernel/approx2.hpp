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

float a_inverse[] = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     -3, 3, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      2,-2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-2,-1, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 1, 1, 0, 0,
                     -3, 0, 3, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0,
                      0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0,-2, 0,-1, 0,
                      9,-9,-9, 9, 6, 3,-6,-3, 6,-6, 3,-3, 4, 2, 2, 1,
                     -6, 6, 6,-6,-3,-3, 3, 3,-4, 4,-2, 2,-2,-2,-1,-1,
                      2, 0,-2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, 1, 0, 1, 0,
                     -6, 6, 6,-6,-4,-2, 4, 2,-3, 3,-3, 3,-2,-1,-2,-1,
                      4,-4,-4, 4, 2, 2,-2,-2, 2,-2, 2,-2, 1, 1, 1, 1 };

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
        if (x < 0 || y < 0 || idims[0] < x+1 || idims[1] < y+1) { //check index in bounds
            gFlag = true;
        }

        dim_t const grid_x = floor(x),   grid_y = floor(y);   // nearest grid
        LocT const off_x  = x - grid_x, off_y  = y - grid_y; // fractional offset

        // Check if pVal - 1, pVal, and pVal + 1 are both valid indices
        bool condY = (y > 0 && y < idims[1] - 2);
        bool condX = (x > 0 && x < idims[0] - 2);

        // Compute weights used
        //f0x = (in[ioff + 1] - in[ioff])/(InT)2 + (in[ioff] - in[ioff - 1])/(InT)2 
        InT zero = scalar<InT>(0);

        dim_t const omId = idw * ostrides[3] + idz * ostrides[2]
                         + idy * ostrides[1] + idx;
        if(gFlag) {
            out[omId] = scalar<InT>(offGrid);
        } else {
            dim_t ioff = idw * istrides[3] + idz * istrides[2]
                    + grid_y * istrides[1] + grid_x;
            InT f00 = in[ioff];
            InT f10 = in[ioff + 1];
            InT f01 = in[ioff + istrides[1]];
            InT f11 = in[ioff + istrides[1] + 1];
            InT f00x  = ((in[ioff + 1] - in[ioff - 1]))*scalar<InT>(0.5);
            InT f10x  = ((in[ioff + 2] - in[ioff]))*scalar<InT>(0.5);
            InT f01x  = ((in[ioff + istrides[1] + 1] - in[ioff + istrides[1] - 1]))*scalar<InT>(0.5);
            InT f11x  = ((in[ioff + istrides[1] + 2] - in[ioff + istrides[1]]))*scalar<InT>(0.5);
            InT f00y  = 0; //((in[ioff + istrides[1]] - in[ioff - istrides[1]]))*scalar<InT>(0.5);
            InT f10y  = 0; //((in[ioff + istrides[1] + 1] - in[ioff - istrides[1] + 1]))*scalar<InT>(0.5);
            InT f01y  = ((in[ioff + 2*istrides[1]] - in[ioff]))*scalar<InT>(0.5);
            InT f11y  = ((in[ioff + 2*istrides[1] + 1] - in[ioff + 1]))*scalar<InT>(0.5);

            InT f00xy = 0; //(in[ioff + istrides[1] + 1] - in[ioff - istrides[1] + 1] - in[ioff + istrides[1] - 1] + in[ioff - istrides[1] - 1])*scalar<InT>(0.25);
            InT f10xy = 0; //(in[ioff + istrides[1] + 2] - in[ioff - istrides[1] + 2] - in[ioff + istrides[1]] + in[ioff - istrides[1]])*scalar<InT>(0.25);
            InT f01xy = (in[ioff + 2*istrides[1] + 1] - in[ioff + 1] - in[ioff + 2*istrides[1] - 1] + in[ioff - 1])*scalar<InT>(0.25);
            InT f11xy = (in[ioff + 2*istrides[1] + 2] - in[ioff + 2] - in[ioff + 2*istrides[1]] + in[ioff])*scalar<InT>(0.25);

            InT x[] = { f00, f10, f01, f11, f00x, f10x, f01x, f11x, f00y, f10y, f01y, f11y, f00xy, f10xy, f01xy, f11xy };
            af::dim4  d(16, 1,1,1);
            af::dim4 d2(16,16,1,1);

            Array<InT> ax = createHostDataArray(d, x);
            Array<InT> aAi = castArray<InT>(getHandle(createHostDataArray(d2, a_inverse)));
            //Array<InT> aAi = createHostDataArray(d2, a_inverse);
            //af_print_array(getHandle(ax));
            //af_print_array(getHandle(aAi));

            Array<InT> a_consts = matmul(aAi, ax, AF_MAT_TRANS, AF_MAT_NONE);
            //af_print_array(getHandle(a_consts));
            InT * a = a_consts.get();

            InT x2 = off_x * off_x;
            InT x3 = x2 * off_x;
            InT y2 = off_y * off_y;
            InT y3 = y2 * off_y;

            // Write Final Value
            out[omId] = (a[0] + a[4] * off_y + a[8]  * y2 + a[12] * y3) +
                        (a[1] + a[5] * off_y + a[9]  * y2 + a[13] * y3) * off_x +
                        (a[2] + a[6] * off_y + a[10] * y2 + a[14] * y3) * x2 +
                        (a[3] + a[6] * off_y + a[11] * y2 + a[15] * y3) * x3;
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
