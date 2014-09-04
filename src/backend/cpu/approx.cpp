#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <approx.hpp>
#include <stdexcept>

namespace cpu
{
    ///////////////////////////////////////////////////////////////////////////
    // Approx1
    ///////////////////////////////////////////////////////////////////////////
    template<typename Ty, typename Tp, af_interp_type method>
    struct approx1_op
    {
        void operator()(Ty *out, const af::dim4 &odims, const dim_type oElems,
                  const Ty *in,  const af::dim4 &idims, const dim_type iElems,
                  const Tp *pos, const af::dim4 &pdims,
                  const af::dim4 &ostrides, const af::dim4 &istrides, const af::dim4 &pstrides,
                  const float offGrid, const dim_type idx)
        {
            return;
        }
    };

    template<typename Ty, typename Tp>
    struct approx1_op<Ty, Tp, AF_INTERP_NEAREST>
    {
        void operator()(Ty *out, const af::dim4 &odims, const dim_type oElems,
                  const Ty *in,  const af::dim4 &idims, const dim_type iElems,
                  const Tp *pos, const af::dim4 &pdims,
                  const af::dim4 &ostrides, const af::dim4 &istrides, const af::dim4 &pstrides,
                  const float offGrid, const dim_type idx)
        {
            const dim_type pmId = idx;

            const Tp x = pos[pmId];
            bool gFlag = false;
            if (x < 0 || idims[0] < x+1) {
                gFlag = true;
            }

            for(dim_type idw = 0; idw < odims[3]; idw++) {
                for(dim_type idz = 0; idz < odims[2]; idz++) {
                    for(dim_type idy = 0; idy < odims[1]; idy++) {
                        const dim_type omId = idw * ostrides[3] + idz * ostrides[2]
                                            + idy * ostrides[1] + idx;
                        if(gFlag) {
                            out[omId] = constant<Ty>(offGrid);
                        } else {
                            dim_type ioff = idw * istrides[3] + idz * istrides[2]
                                          + idy * istrides[1];
                            const dim_type iMem = round(x) + ioff;

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
        void operator()(Ty *out, const af::dim4 &odims, const dim_type oElems,
                  const Ty *in,  const af::dim4 &idims, const dim_type iElems,
                  const Tp *pos, const af::dim4 &pdims,
                  const af::dim4 &ostrides, const af::dim4 &istrides, const af::dim4 &pstrides,
                  const float offGrid, const dim_type idx)
        {
            const dim_type pmId = idx;

            const Tp x = pos[pmId];
            bool gFlag = false;
            if (x < 0 || idims[0] < x+1) {
                gFlag = true;
            }

            const Tp grix = floor(x);  // nearest grid
            const Tp off_x = x - grix; // fractional offset

            for(dim_type idw = 0; idw < odims[3]; idw++) {
                for(dim_type idz = 0; idz < odims[2]; idz++) {
                    for(dim_type idy = 0; idy < odims[1]; idy++) {
                        const dim_type omId = idw * ostrides[3] + idz * ostrides[2]
                                            + idy * ostrides[1] + idx;
                        if(gFlag) {
                            out[omId] = constant<Ty>(offGrid);
                        } else {
                            Tp w = 0;
                            Ty y = constant<Ty>(0);
                            dim_type ioff = idw * istrides[3] + idz * istrides[2]
                                          + idy * istrides[1];
                            for(dim_type xx = 0; xx <= (x < idims[0] - 1); ++xx) {
                                Tp fxx = (Tp)(xx);
                                Tp wx = 1 - fabs(off_x - fxx);
                                dim_type imId = (dim_type)(fxx + grix) + ioff;
                                Ty yt = in[imId];
                                y = y + (yt * wx);
                                w = w + wx;
                            }
                            out[omId] = (y / w);
                        }
                    }
                }
            }
        }
    };

    template<typename Ty, typename Tp, af_interp_type method>
    void approx1_(Ty *out, const af::dim4 &odims, const dim_type oElems,
            const Ty *in,  const af::dim4 &idims, const dim_type iElems,
            const Tp *pos, const af::dim4 &pdims,
            const af::dim4 &ostrides, const af::dim4 &istrides, const af::dim4 &pstrides,
            const float offGrid)
    {
        approx1_op<Ty, Tp, method> op;
        for(dim_type x = 0; x < odims[0]; x++) {
            op(out, odims, oElems, in, idims, iElems, pos, pdims,
               ostrides, istrides, pstrides, offGrid, x);
        }
    }

    template<typename Ty, typename Tp>
    Array<Ty> *approx1(const Array<Ty> &in, const Array<Tp> &pos,
                       const af_interp_type method, const float offGrid)
    {
        af::dim4 odims = in.dims();
        odims[0] = pos.dims()[0];

        // Create output placeholder
        Array<Ty> *out = createEmptyArray<Ty>(odims);

        switch(method) {
            case AF_INTERP_NEAREST:
                approx1_<Ty, Tp, AF_INTERP_NEAREST>
                        (out->get(), out->dims(), out->elements(),
                         in.get(), in.dims(), in.elements(), pos.get(), pos.dims(),
                         out->strides(), in.strides(), pos.strides(), offGrid);
                break;
            case AF_INTERP_LINEAR:
                approx1_<Ty, Tp, AF_INTERP_LINEAR>
                        (out->get(), out->dims(), out->elements(),
                         in.get(), in.dims(), in.elements(), pos.get(), pos.dims(),
                         out->strides(), in.strides(), pos.strides(), offGrid);
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
        void operator()(Ty *out, const af::dim4 &odims, const dim_type oElems,
                  const Ty *in,  const af::dim4 &idims, const dim_type iElems,
                  const Tp *pos, const af::dim4 &pdims, const Tp *qos, const af::dim4 &qdims,
                  const af::dim4 &ostrides, const af::dim4 &istrides,
                  const af::dim4 &pstrides, const af::dim4 &qstrides,
                  const float offGrid, const dim_type idx, const dim_type idy)
        {
            return;
        }
    };

    template<typename Ty, typename Tp>
    struct approx2_op<Ty, Tp, AF_INTERP_NEAREST>
    {
        void operator()(Ty *out, const af::dim4 &odims, const dim_type oElems,
                  const Ty *in,  const af::dim4 &idims, const dim_type iElems,
                  const Tp *pos, const af::dim4 &pdims, const Tp *qos, const af::dim4 &qdims,
                  const af::dim4 &ostrides, const af::dim4 &istrides,
                  const af::dim4 &pstrides, const af::dim4 &qstrides,
                  const float offGrid, const dim_type idx, const dim_type idy)
        {
            const dim_type pmId = idy * pstrides[1] + idx;
            const dim_type qmId = idy * qstrides[1] + idx;

            bool gFlag = false;
            const Tp x = pos[pmId], y = qos[qmId];
            if (x < 0 || y < 0 || idims[0] < x+1 || idims[1] < y+1) {
                gFlag = true;
            }

            for(dim_type idw = 0; idw < odims[3]; idw++) {
                for(dim_type idz = 0; idz < odims[2]; idz++) {
                    const dim_type omId = idw * ostrides[3] + idz * ostrides[2]
                                        + idy * ostrides[1] + idx;
                    if(gFlag) {
                        out[omId] = constant<Ty>(offGrid);
                    } else {
                        const dim_type grid_x = round(x), grid_y = round(y); // nearest grid
                        const dim_type imId = idw * istrides[3] +
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
        void operator()(Ty *out, const af::dim4 &odims, const dim_type oElems,
                  const Ty *in,  const af::dim4 &idims, const dim_type iElems,
                  const Tp *pos, const af::dim4 &pdims, const Tp *qos, const af::dim4 &qdims,
                  const af::dim4 &ostrides, const af::dim4 &istrides,
                  const af::dim4 &pstrides, const af::dim4 &qstrides,
                  const float offGrid, const dim_type idx, const dim_type idy)
        {
            const dim_type pmId = idy * pstrides[1] + idx;
            const dim_type qmId = idy * qstrides[1] + idx;

            bool gFlag = false;
            const Tp x = pos[pmId], y = qos[qmId];
            if (x < 0 || y < 0 || idims[0] < x+1 || idims[1] < y+1) {
                gFlag = true;
            }

            for(dim_type idw = 0; idw < odims[3]; idw++) {
                for(dim_type idz = 0; idz < odims[2]; idz++) {
                    const dim_type omId = idw * ostrides[3] + idz * ostrides[2]
                                        + idy * ostrides[1] + idx;
                    if(gFlag) {
                        out[omId] = constant<Ty>(offGrid);
                    } else {
                        const Tp grid_x = floor(x),   grid_y = floor(y);   // nearest grid
                        const Tp off_x  = x - grid_x, off_y  = y - grid_y; // fractional offset

                        Tp w = 0;
                        Ty z = constant<Ty>(0);
                        dim_type ioff = idw * istrides[3] + idz * istrides[2];
                        for(dim_type yy = 0; yy <= (y < idims[1] - 1); ++yy) {
                            Tp fyy = (Tp)(yy);
                            Tp wy = 1 - fabs(off_y - fyy);
                            dim_type idyy = (dim_type)(fyy + grid_y);
                            for(dim_type xx = 0; xx <= (x < idims[0] - 1); ++xx) {
                                Tp fxx = (Tp)(xx);
                                Tp wxy = (1 - fabs(off_x - fxx)) * wy;
                                dim_type imId = idyy * istrides[1] + (dim_type)(fxx + grid_x) + ioff;
                                Ty zt = in[imId];
                                z = z + (zt * wxy);
                                w = w + wxy;
                            }
                        }
                        out[omId] = z / w;
                    }
                }
            }
        }
    };

    template<typename Ty, typename Tp, af_interp_type method>
    void approx2_(Ty *out, const af::dim4 &odims, const dim_type oElems,
            const Ty *in,  const af::dim4 &idims, const dim_type iElems,
            const Tp *pos, const af::dim4 &pdims, const Tp *qos, const af::dim4 &qdims,
            const af::dim4 &ostrides, const af::dim4 &istrides,
            const af::dim4 &pstrides, const af::dim4 &qstrides,
            const float offGrid)
    {
        approx2_op<Ty, Tp, method> op;
        for(dim_type y = 0; y < odims[1]; y++) {
            for(dim_type x = 0; x < odims[0]; x++) {
                op(out, odims, oElems, in, idims, iElems, pos, pdims, qos, qdims,
                    ostrides, istrides, pstrides, qstrides, offGrid, x, y);
            }
        }
    }

    template<typename Ty, typename Tp>
    Array<Ty> *approx2(const Array<Ty> &in, const Array<Tp> &pos0, const Array<Tp> &pos1,
                       const af_interp_type method, const float offGrid)
    {
        af::dim4 odims = in.dims();
        odims[0] = pos0.dims()[0];
        odims[1] = pos0.dims()[1];

        // Create output placeholder
        Array<Ty> *out = createEmptyArray<Ty>(odims);

        switch(method) {
            case AF_INTERP_NEAREST:
                approx2_<Ty, Tp, AF_INTERP_NEAREST>
                        (out->get(), out->dims(), out->elements(),
                         in.get(), in.dims(), in.elements(),
                         pos0.get(), pos0.dims(), pos1.get(), pos1.dims(),
                         out->strides(), in.strides(), pos0.strides(), pos1.strides(),
                         offGrid);
                break;
            case AF_INTERP_LINEAR:
                approx2_<Ty, Tp, AF_INTERP_LINEAR>
                        (out->get(), out->dims(), out->elements(),
                         in.get(), in.dims(), in.elements(),
                         pos0.get(), pos0.dims(), pos1.get(), pos1.dims(),
                         out->strides(), in.strides(), pos0.strides(), pos1.strides(),
                         offGrid);
                break;
            default:
                break;
        }
        return out;
    }

#define INSTANTIATE(Ty, Tp)                                                                     \
    template Array<Ty>* approx1<Ty, Tp>(const Array<Ty> &in, const Array<Tp> &pos,              \
                                        const af_interp_type method, const float offGrid);      \
    template Array<Ty>* approx2<Ty, Tp>(const Array<Ty> &in, const Array<Tp> &pos0,             \
                                        const Array<Tp> &pos1, const af_interp_type method,     \
                                        const float offGrid);                                   \

    INSTANTIATE(float  , float )
    INSTANTIATE(double , double)
    INSTANTIATE(cfloat , float )
    INSTANTIATE(cdouble, double)
}

