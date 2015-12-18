/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <math.hpp>

namespace cpu
{
namespace kernel
{

using af::dim4;

template<typename Ty, typename Tp, af_interp_type method>
struct approx1_op
{
    void operator()(Ty *out, const af::dim4 &odims, const dim_t oElems,
              const Ty *in,  const af::dim4 &idims, const dim_t iElems,
              const Tp *pos, const af::dim4 &pdims,
              const af::dim4 &ostrides, const af::dim4 &istrides, const af::dim4 &pstrides,
              const float offGrid, const bool pBatch,
              const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw)
    {
        return;
    }
};

template<typename Ty, typename Tp>
struct approx1_op<Ty, Tp, AF_INTERP_NEAREST>
{
    void operator()(Ty *out, const af::dim4 &odims, const dim_t oElems,
              const Ty *in,  const af::dim4 &idims, const dim_t iElems,
              const Tp *pos, const af::dim4 &pdims,
              const af::dim4 &ostrides, const af::dim4 &istrides, const af::dim4 &pstrides,
              const float offGrid, const bool pBatch,
              const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw)
    {
        dim_t pmId = idx;
        if(pBatch) pmId += idw * pstrides[3] + idz * pstrides[2] + idy * pstrides[1];

        const Tp x = pos[pmId];
        bool gFlag = false;
        if (x < 0 || idims[0] < x+1) {  // No need to check y
            gFlag = true;
        }

        const dim_t omId = idw * ostrides[3] + idz * ostrides[2]
                         + idy * ostrides[1] + idx;
        if(gFlag) {
            out[omId] = scalar<Ty>(offGrid);
        } else {
            dim_t ioff = idw * istrides[3] + idz * istrides[2]
                       + idy * istrides[1];
            const dim_t iMem = round(x) + ioff;

            out[omId] = in[iMem];
        }
    }
};

template<typename Ty, typename Tp>
struct approx1_op<Ty, Tp, AF_INTERP_LINEAR>
{
    void operator()(Ty *out, const af::dim4 &odims, const dim_t oElems,
              const Ty *in,  const af::dim4 &idims, const dim_t iElems,
              const Tp *pos, const af::dim4 &pdims,
              const af::dim4 &ostrides, const af::dim4 &istrides, const af::dim4 &pstrides,
              const float offGrid, const bool pBatch,
              const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw)
    {
        dim_t pmId = idx;
        if(pBatch) pmId += idw * pstrides[3] + idz * pstrides[2] + idy * pstrides[1];

        const Tp x = pos[pmId];
        bool gFlag = false;
        if (x < 0 || idims[0] < x+1) {
            gFlag = true;
        }

        const dim_t grid_x = floor(x);  // nearest grid
        const Tp off_x = x - grid_x; // fractional offset

        const dim_t omId = idw * ostrides[3] + idz * ostrides[2]
                         + idy * ostrides[1] + idx;
        if(gFlag) {
            out[omId] = scalar<Ty>(offGrid);
        } else {
            dim_t ioff = idw * istrides[3] + idz * istrides[2] + idy * istrides[1] + grid_x;

            // Check if x and x + 1 are both valid indices
            bool cond = (x < idims[0] - 1);
            // Compute Left and Right Weighted Values
            Ty yl = ((Tp)1.0 - off_x) * in[ioff];
            Ty yr = cond ? (off_x) * in[ioff + 1] : scalar<Ty>(0);
            Ty yo = yl + yr;
            // Compute Weight used
            Tp wt = cond ? (Tp)1.0 : (Tp)(1.0 - off_x);
            // Write final value
            out[omId] = (yo / wt);
        }
    }
};

template<typename Ty, typename Tp, af_interp_type method>
void approx1(Array<Ty> output, Array<Ty> const input,
             Array<Tp> const position, float const offGrid)
{
    Ty * out = output.get();
    Ty const * const in  = input.get();
    Tp const * const pos = position.get();
    dim4 const odims     = output.dims();
    dim4 const idims     = input.dims();
    dim4 const pdims     = position.dims();
    dim4 const ostrides  = output.strides();
    dim4 const istrides  = input.strides();
    dim4 const pstrides  = position.strides();
    dim_t const oElems   = output.elements();
    dim_t const iElems   = input.elements();

    approx1_op<Ty, Tp, method> op;
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
