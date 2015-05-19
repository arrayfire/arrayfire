/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <approx.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cpu.hpp>

namespace cpu
{
    ///////////////////////////////////////////////////////////////////////////
    // Approx1
    ///////////////////////////////////////////////////////////////////////////
    template<typename Ty, typename Tp, af_interp_type method>
    struct approx1_op
    {
        void operator()(Ty *out, const af::dim4 &odims, const dim_t oElems,
                  const Ty *in,  const af::dim4 &idims, const dim_t iElems,
                  const Tp *pos, const af::dim4 &pdims,
                  const af::dim4 &ostrides, const af::dim4 &istrides, const af::dim4 &pstrides,
                  const float offGrid, const dim_t idx)
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
                  const float offGrid, const dim_t idx)
        {
            const dim_t pmId = idx;

            const Tp x = pos[pmId];
            bool gFlag = false;
            if (x < 0 || idims[0] < x+1) {
                gFlag = true;
            }

            for(dim_t idw = 0; idw < odims[3]; idw++) {
                for(dim_t idz = 0; idz < odims[2]; idz++) {
                    for(dim_t idy = 0; idy < odims[1]; idy++) {
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
                }
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
                  const float offGrid, const dim_t idx)
        {
            const dim_t pmId = idx;

            const Tp x = pos[pmId];
            bool gFlag = false;
            if (x < 0 || idims[0] < x+1) {
                gFlag = true;
            }

            const Tp grid_x = floor(x);  // nearest grid
            const Tp off_x = x - grid_x; // fractional offset

            for(dim_t idw = 0; idw < odims[3]; idw++) {
                for(dim_t idz = 0; idz < odims[2]; idz++) {
                    for(dim_t idy = 0; idy < odims[1]; idy++) {
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
                }
            }
        }
    };

    template<typename Ty, typename Tp, af_interp_type method>
    void approx1_(Ty *out, const af::dim4 &odims, const dim_t oElems,
            const Ty *in,  const af::dim4 &idims, const dim_t iElems,
            const Tp *pos, const af::dim4 &pdims,
            const af::dim4 &ostrides, const af::dim4 &istrides, const af::dim4 &pstrides,
            const float offGrid)
    {
        approx1_op<Ty, Tp, method> op;
        for(dim_t x = 0; x < odims[0]; x++) {
            op(out, odims, oElems, in, idims, iElems, pos, pdims,
               ostrides, istrides, pstrides, offGrid, x);
        }
    }

    template<typename Ty, typename Tp>
    Array<Ty> approx1(const Array<Ty> &in, const Array<Tp> &pos,
                       const af_interp_type method, const float offGrid)
    {
        af::dim4 odims = in.dims();
        odims[0] = pos.dims()[0];

        // Create output placeholder
        Array<Ty> out = createEmptyArray<Ty>(odims);

        switch(method) {
            case AF_INTERP_NEAREST:
                approx1_<Ty, Tp, AF_INTERP_NEAREST>
                        (out.get(), out.dims(), out.elements(),
                         in.get(), in.dims(), in.elements(), pos.get(), pos.dims(),
                         out.strides(), in.strides(), pos.strides(), offGrid);
                break;
            case AF_INTERP_LINEAR:
                approx1_<Ty, Tp, AF_INTERP_LINEAR>
                        (out.get(), out.dims(), out.elements(),
                         in.get(), in.dims(), in.elements(), pos.get(), pos.dims(),
                         out.strides(), in.strides(), pos.strides(), offGrid);
                break;
            default:
                break;
        }
        return out;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Approx2
    ///////////////////////////////////////////////////////////////////////////
    template<typename Ty, typename Tp, af_interp_type method>
    struct approx2_op
    {
        void operator()(Ty *out, const af::dim4 &odims, const dim_t oElems,
                  const Ty *in,  const af::dim4 &idims, const dim_t iElems,
                  const Tp *pos, const af::dim4 &pdims, const Tp *qos, const af::dim4 &qdims,
                  const af::dim4 &ostrides, const af::dim4 &istrides,
                  const af::dim4 &pstrides, const af::dim4 &qstrides,
                  const float offGrid, const dim_t idx, const dim_t idy)
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
                  const float offGrid, const dim_t idx, const dim_t idy)
        {
            const dim_t pmId = idy * pstrides[1] + idx;
            const dim_t qmId = idy * qstrides[1] + idx;

            bool gFlag = false;
            const Tp x = pos[pmId], y = qos[qmId];
            if (x < 0 || y < 0 || idims[0] < x+1 || idims[1] < y+1) {
                gFlag = true;
            }

            for(dim_t idw = 0; idw < odims[3]; idw++) {
                for(dim_t idz = 0; idz < odims[2]; idz++) {
                    const dim_t omId = idw * ostrides[3] + idz * ostrides[2]
                                        + idy * ostrides[1] + idx;
                    if(gFlag) {
                        out[omId] = scalar<Ty>(offGrid);
                    } else {
                        const dim_t grid_x = round(x), grid_y = round(y); // nearest grid
                        const dim_t imId = idw * istrides[3] +
                                              idz * istrides[2] +
                                              grid_y * istrides[1] + grid_x;
                        out[omId] = in[imId];
                    }
                }
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
                  const float offGrid, const dim_t idx, const dim_t idy)
        {
            const dim_t pmId = idy * pstrides[1] + idx;
            const dim_t qmId = idy * qstrides[1] + idx;

            bool gFlag = false;
            const Tp x = pos[pmId], y = qos[qmId];
            if (x < 0 || y < 0 || idims[0] < x+1 || idims[1] < y+1) {
                gFlag = true;
            }

            const Tp grid_x = floor(x),   grid_y = floor(y);   // nearest grid
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

            for(dim_t idw = 0; idw < odims[3]; idw++) {
                for(dim_t idz = 0; idz < odims[2]; idz++) {
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
                        Ty y01 = (condX) ?          wt01 * in[ioff + 1]                   : zero;
                        Ty y11 = (condX && condY) ? wt11 * in[ioff + istrides[1] + 1] : zero;

                        Ty yo = y00 + y10 + y01 + y11;

                        // Write Final Value
                        out[omId] = (yo / wt);

                    }
                }
            }
        }
    };

    template<typename Ty, typename Tp, af_interp_type method>
    void approx2_(Ty *out, const af::dim4 &odims, const dim_t oElems,
            const Ty *in,  const af::dim4 &idims, const dim_t iElems,
            const Tp *pos, const af::dim4 &pdims, const Tp *qos, const af::dim4 &qdims,
            const af::dim4 &ostrides, const af::dim4 &istrides,
            const af::dim4 &pstrides, const af::dim4 &qstrides,
            const float offGrid)
    {
        approx2_op<Ty, Tp, method> op;
        for(dim_t y = 0; y < odims[1]; y++) {
            for(dim_t x = 0; x < odims[0]; x++) {
                op(out, odims, oElems, in, idims, iElems, pos, pdims, qos, qdims,
                    ostrides, istrides, pstrides, qstrides, offGrid, x, y);
            }
        }
    }

    template<typename Ty, typename Tp>
    Array<Ty> approx2(const Array<Ty> &in, const Array<Tp> &pos0, const Array<Tp> &pos1,
                       const af_interp_type method, const float offGrid)
    {
        af::dim4 odims = in.dims();
        odims[0] = pos0.dims()[0];
        odims[1] = pos0.dims()[1];

        // Create output placeholder
        Array<Ty> out = createEmptyArray<Ty>(odims);

        switch(method) {
            case AF_INTERP_NEAREST:
                approx2_<Ty, Tp, AF_INTERP_NEAREST>
                        (out.get(), out.dims(), out.elements(),
                         in.get(), in.dims(), in.elements(),
                         pos0.get(), pos0.dims(), pos1.get(), pos1.dims(),
                         out.strides(), in.strides(), pos0.strides(), pos1.strides(),
                         offGrid);
                break;
            case AF_INTERP_LINEAR:
                approx2_<Ty, Tp, AF_INTERP_LINEAR>
                        (out.get(), out.dims(), out.elements(),
                         in.get(), in.dims(), in.elements(),
                         pos0.get(), pos0.dims(), pos1.get(), pos1.dims(),
                         out.strides(), in.strides(), pos0.strides(), pos1.strides(),
                         offGrid);
                break;
            default:
                break;
        }
        return out;
    }

#define INSTANTIATE(Ty, Tp)                                                                     \
    template Array<Ty> approx1<Ty, Tp>(const Array<Ty> &in, const Array<Tp> &pos,              \
                                        const af_interp_type method, const float offGrid);      \
    template Array<Ty> approx2<Ty, Tp>(const Array<Ty> &in, const Array<Tp> &pos0,             \
                                        const Array<Tp> &pos1, const af_interp_type method,     \
                                        const float offGrid);                                   \

    INSTANTIATE(float  , float )
    INSTANTIATE(double , double)
    INSTANTIATE(cfloat , float )
    INSTANTIATE(cdouble, double)
}
