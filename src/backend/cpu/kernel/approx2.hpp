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

namespace kernel
{

using af::dim4;
using cpu::scalar;
using cpu::Array;

template<typename Ty, typename Tp, af_interp_type method>
struct approx2_op
{
    void operator()(Ty *out, const af::dim4 &odims, const dim_t oElems,
              const Ty *in,  const af::dim4 &idims, const dim_t iElems,
              const Tp *pos, const af::dim4 &pdims, const Tp *qos, const af::dim4 &qdims,
              const af::dim4 &ostrides, const af::dim4 &istrides,
              const af::dim4 &pstrides, const af::dim4 &qstrides,
              const float offGrid, const bool pBatch,
              const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw)
    {
        return;
    }
};

template<typename Ty, typename Tp>
struct approx2_op<Ty, Tp, AF_INTERP_NEAREST>
{
    void operator()(Ty *out, const af::dim4 &odims, const dim_t oElems,
              const Ty *in,  const af::dim4 &idims, const dim_t iElems,
              const Tp *pos, const af::dim4 &pdims, const Tp *qos, const af::dim4 &qdims,
              const af::dim4 &ostrides, const af::dim4 &istrides,
              const af::dim4 &pstrides, const af::dim4 &qstrides,
              const float offGrid, const bool pBatch,
              const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw)
    {
        dim_t pmId = idy * pstrides[1] + idx;
        dim_t qmId = idy * qstrides[1] + idx;
        if(pBatch) {
            pmId += idw * pstrides[3] + idz * pstrides[2];
            qmId += idw * qstrides[3] + idz * qstrides[2];
        }

        bool gFlag = false;
        const Tp x = pos[pmId], y = qos[qmId];
        if (x < 0 || y < 0 || idims[0] < x+1 || idims[1] < y+1) {
            gFlag = true;
        }

        const dim_t omId = idw * ostrides[3] + idz * ostrides[2]
                         + idy * ostrides[1] + idx;
        if(gFlag) {
            out[omId] = scalar<Ty>(offGrid);
        } else {
            const dim_t grid_x = round(x), grid_y = round(y); // nearest grid
            const dim_t imId = idw * istrides[3] + idz * istrides[2] +
                            grid_y * istrides[1] + grid_x;
            out[omId] = in[imId];
        }
    }
};

template<typename Ty, typename Tp>
struct approx2_op<Ty, Tp, AF_INTERP_LINEAR>
{
    void operator()(Ty *out, const af::dim4 &odims, const dim_t oElems,
              const Ty *in,  const af::dim4 &idims, const dim_t iElems,
              const Tp *pos, const af::dim4 &pdims, const Tp *qos, const af::dim4 &qdims,
              const af::dim4 &ostrides, const af::dim4 &istrides,
              const af::dim4 &pstrides, const af::dim4 &qstrides,
              const float offGrid, const bool pBatch,
              const dim_t idx, const dim_t idy, const dim_t idz, const dim_t idw)
    {
        dim_t pmId = idy * pstrides[1] + idx;
        dim_t qmId = idy * qstrides[1] + idx;
        if(pBatch) {
            pmId += idw * pstrides[3] + idz * pstrides[2];
            qmId += idw * qstrides[3] + idz * qstrides[2];
        }

        bool gFlag = false;
        const Tp x = pos[pmId], y = qos[qmId];
        if (x < 0 || y < 0 || idims[0] < x+1 || idims[1] < y+1) {
            gFlag = true;
        }

        const dim_t grid_x = floor(x),   grid_y = floor(y);   // nearest grid
        const Tp off_x  = x - grid_x, off_y  = y - grid_y; // fractional offset

        // Check if pVal and pVal + 1 are both valid indices
        bool condY = (y < idims[1] - 1);
        bool condX = (x < idims[0] - 1);

        // Compute wieghts used
        Tp wt00 = ((Tp)1.0 - off_x) * ((Tp)1.0 - off_y);
        Tp wt10 = (condY) ? ((Tp)1.0 - off_x) * (off_y) : 0;
        Tp wt01 = (condX) ? (off_x) * ((Tp)1.0 - off_y) : 0;
        Tp wt11 = (condX && condY) ? (off_x) * (off_y)  : 0;

        Tp wt = wt00 + wt10 + wt01 + wt11;
        Ty zero = scalar<Ty>(0);

        const dim_t omId = idw * ostrides[3] + idz * ostrides[2]
                         + idy * ostrides[1] + idx;
        if(gFlag) {
            out[omId] = scalar<Ty>(offGrid);
        } else {
            dim_t ioff = idw * istrides[3] + idz * istrides[2]
                    + grid_y * istrides[1] + grid_x;

            // Compute Weighted Values
            Ty y00 =                    wt00 * in[ioff];
            Ty y10 = (condY) ?          wt10 * in[ioff + istrides[1]]     : zero;
            Ty y01 = (condX) ?          wt01 * in[ioff + 1]               : zero;
            Ty y11 = (condX && condY) ? wt11 * in[ioff + istrides[1] + 1] : zero;

            Ty yo = y00 + y10 + y01 + y11;

            // Write Final Value
            out[omId] = (yo / wt);
        }
    }
};

template<typename Ty, typename Tp, af_interp_type method>
void approx2(Array<Ty> output, Array<Ty> const input,
             Array<Tp> const position, Array<Tp> const qosition,
             float const offGrid)
{
    Ty * out = output.get();
    Ty const * const in  = input.get();
    Tp const * const pos = position.get();
    Tp const * const qos = qosition.get();
    dim4 const odims     = output.dims();
    dim4 const idims     = input.dims();
    dim4 const pdims     = position.dims();
    dim4 const qdims     = qosition.dims();
    dim4 const ostrides  = output.strides();
    dim4 const istrides  = input.strides();
    dim4 const pstrides  = position.strides();
    dim4 const qstrides  = qosition.strides();
    dim_t const oElems   = output.elements();
    dim_t const iElems   = input.elements();

    approx2_op<Ty, Tp, method> op;
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
