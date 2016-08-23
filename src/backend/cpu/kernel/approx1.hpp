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

namespace cpu
{
namespace kernel
{

template<typename InT, typename LocT, af_interp_type Method>
struct approx1_op
{
    void operator()(InT *out, af::dim4 const & odims, dim_t const oElems,
              InT const * const in,  af::dim4 const & idims, dim_t const iElems,
              LocT const * const pos, af::dim4 const & pdims,
              af::dim4 const & ostrides, af::dim4 const & istrides, af::dim4 const & pstrides,
              float const offGrid, bool const pBatch,
              dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        return;
    }
};

template<typename InT, typename LocT>
struct approx1_op<InT, LocT, AF_INTERP_NEAREST>
{
    void operator()(InT *out, af::dim4 const & odims, dim_t const oElems,
              InT const * const in,  af::dim4 const & idims, dim_t const iElems,
              LocT const * const pos, af::dim4 const & pdims,
              af::dim4 const & ostrides, af::dim4 const & istrides, af::dim4 const & pstrides,
              float const offGrid, bool const pBatch,
              dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        dim_t pmId = idx;
        if(pBatch) pmId += idw * pstrides[3] + idz * pstrides[2] + idy * pstrides[1];

        LocT const x = pos[pmId];
        bool gFlag = false;
        if (x < 0 || idims[0] < x+1) {  // No need to check y
            gFlag = true;
        }

        dim_t const omId = idw * ostrides[3] + idz * ostrides[2]
                         + idy * ostrides[1] + idx;
        if(gFlag) {
            out[omId] = scalar<InT>(offGrid);
        } else {
            dim_t ioff = idw * istrides[3] + idz * istrides[2]
                       + idy * istrides[1];
            dim_t const iMem = round(x) + ioff;

            out[omId] = in[iMem];
        }
    }
};

template<typename InT, typename LocT>
struct approx1_op<InT, LocT, AF_INTERP_LINEAR>
{
    void operator()(InT *out, af::dim4 const & odims, dim_t const oElems,
              InT const * const in,  af::dim4 const & idims, dim_t const iElems,
              LocT const * const pos, af::dim4 const & pdims,
              af::dim4 const & ostrides, af::dim4 const & istrides, af::dim4 const & pstrides,
              float const offGrid, bool const pBatch,
              dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        dim_t pmId = idx;
        if(pBatch) pmId += idw * pstrides[3] + idz * pstrides[2] + idy * pstrides[1];

        LocT const x = pos[pmId];
        bool gFlag = false;
        if (x < 0 || idims[0] < x+1) {
            gFlag = true;
        }

        dim_t const grid_x = floor(x);  // nearest grid
        LocT const off_x = x - grid_x;  // fractional offset

        dim_t const omId = idw * ostrides[3] + idz * ostrides[2]
                         + idy * ostrides[1] + idx;
        if(gFlag) {
            out[omId] = scalar<InT>(offGrid);
        } else {
            dim_t ioff = idw * istrides[3] + idz * istrides[2] + idy * istrides[1] + grid_x;

            // Check if x and x + 1 are both valid indices
            bool cond = (x < idims[0] - 1);
            // Compute Left and Right Weighted Values
            InT yl = ((LocT)1.0 - off_x) * in[ioff];
            InT yr = cond ? (off_x) * in[ioff + 1] : scalar<InT>(0);
            InT yo = yl + yr;
            // Compute Weight used
            LocT wt = cond ? (LocT)1.0 : (LocT)(1.0 - off_x);
            // Write final value
            out[omId] = (yo / wt);
        }
    }
};

template<typename InT, typename LocT>
struct approx1_op<InT, LocT, AF_INTERP_CUBIC>
{
    void operator()(InT *out, af::dim4 const & odims, dim_t const oElems,
              InT const * const in,  af::dim4 const & idims, dim_t const iElems,
              LocT const * const pos, af::dim4 const & pdims,
              af::dim4 const & ostrides, af::dim4 const & istrides, af::dim4 const & pstrides,
              float const offGrid, bool const pBatch,
              dim_t const idx, dim_t const idy, dim_t const idz, dim_t const idw)
    {
        dim_t pmId = idx;
        if(pBatch) pmId += idw * pstrides[3] + idz * pstrides[2] + idy * pstrides[1];

        LocT const x = pos[pmId];
        bool gFlag = false;
        if (x < 0 || idims[0] < x + 1) { //check index of bounds
            gFlag = true;
        }

        dim_t const grid_x = floor(x);  // nearest grid
        LocT  const off_x  = x - grid_x;  // fractional offset

        dim_t const omId = idw * ostrides[3] + idz * ostrides[2]
                         + idy * ostrides[1] + idx;

        if(gFlag) {
            out[omId] = scalar<InT>(offGrid);
        } else {
            dim_t ioff = idw * istrides[3] + idz * istrides[2] + idy * istrides[1] + grid_x;

            //compute basis function values
            InT h00 = (1 + 2 * off_x) * (1 - off_x) * (1 - off_x);
            InT h10 = off_x * (1 - off_x) * (1 - off_x);
            InT h01 = off_x * off_x * (3 - 2 * off_x);
            InT h11 = off_x * off_x * (off_x - 1);
            // Check if x-1, x, and x+1, x+2 are both valid indices
            bool condr  = (grid_x > 0);
            bool condl1 = (grid_x < idims[0] - 1);
            bool condl2 = (grid_x < idims[0] - 2);
            // Compute Left and Right points and tangents
            InT pl = in[ioff];
            InT pr = condl1  ? in[ioff + 1] : in[ioff];
            InT tl = condr   ? scalar<InT>(0.5) * (in[ioff + 1] - in[ioff - 1]) : (in[ioff + 1] - in[ioff]);
            InT tr = condl2  ? scalar<InT>(0.5) * (in[ioff + 2] - in[ioff]) : (condl1) ? in[ioff + 1] - in[ioff] : (in[ioff] - in[ioff - 1]);

            // Write final value
            out[omId] = h00 * pl + h10 * tl + h01 * pr + h11 * tr;
        }
    }
};

template<typename InT, typename LocT, af_interp_type Method>
void approx1(Array<InT> output, Array<InT> const input,
             Array<LocT> const position, float const offGrid)
{
    InT * out = output.get();
    InT const * const in  = input.get();
    LocT const * const pos = position.get();

    af::dim4 const odims     = output.dims();
    af::dim4 const idims     = input.dims();
    af::dim4 const pdims     = position.dims();
    af::dim4 const ostrides  = output.strides();
    af::dim4 const istrides  = input.strides();
    af::dim4 const pstrides  = position.strides();

    dim_t const oElems = output.elements();
    dim_t const iElems = input.elements();

    approx1_op<InT, LocT, Method> op;
    bool pBatch = !(pdims[1] == 1 && pdims[2] == 1 && pdims[3] == 1);

    for(dim_t w = 0; w < odims[3]; w++) {
        for(dim_t z = 0; z < odims[2]; z++) {
            for(dim_t y = 0; y < odims[1]; y++) {
                for(dim_t x = 0; x < odims[0]; x++) {
                    op(out, odims, oElems, in, idims, iElems, pos, pdims,
                       ostrides, istrides, pstrides, offGrid, pBatch, x, y, z, w);
                }
            }
        }
    }
}

}
}
