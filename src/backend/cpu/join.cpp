/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <join.hpp>
#include <stdexcept>
#include <err_cpu.hpp>

namespace cpu
{
    template<typename Tx, typename Ty, int dim>
    void join_(Tx *out, const Tx *X, const Ty *Y,
               const af::dim4 &odims, const af::dim4 &xdims, const af::dim4 &ydims,
               const af::dim4 &ost, const af::dim4 &xst, const af::dim4 &yst)
    {
        af::dim4 offset;
        offset[0] = (dim == 0) ? xdims[0] : 0;
        offset[1] = (dim == 1) ? xdims[1] : 0;
        offset[2] = (dim == 2) ? xdims[2] : 0;
        offset[3] = (dim == 3) ? xdims[3] : 0;
        for(dim_type ow = 0; ow < xdims[3]; ow++) {
            const dim_type xW = ow * xst[3];
            const dim_type oW = ow * ost[3];

            for(dim_type oz = 0; oz < xdims[2]; oz++) {
                const dim_type xZW = xW + oz * xst[2];
                const dim_type oZW = oW + oz * ost[2];

                for(dim_type oy = 0; oy < xdims[1]; oy++) {
                    const dim_type xYZW = xZW + oy * xst[1];
                    const dim_type oYZW = oZW + oy * ost[1];

                    for(dim_type ox = 0; ox < xdims[0]; ox++) {
                        const dim_type iMem = xYZW + ox;
                        const dim_type oMem = oYZW + ox;
                        out[oMem] = X[iMem];
                    }
                }
            }
        }

        for(dim_type ow = 0; ow < ydims[3]; ow++) {
            const dim_type yW = ow * yst[3];
            const dim_type oW = (ow + offset[3]) * ost[3];

            for(dim_type oz = 0; oz < ydims[2]; oz++) {
                const dim_type yZW = yW + oz * yst[2];
                const dim_type oZW = oW + (offset[2] + oz) * ost[2];

                for(dim_type oy = 0; oy < ydims[1]; oy++) {
                    const dim_type yYZW = yZW + oy * yst[1];
                    const dim_type oYZW = oZW + (offset[1] + oy) * ost[1];

                    for(dim_type ox = 0; ox < ydims[0]; ox++) {
                        const dim_type iMem = yYZW + ox;
                        const dim_type oMem = oYZW + (offset[0] + ox);
                        out[oMem] = Y[iMem];
                    }
                }
            }
        }
    }

    template<typename Tx, typename Ty>
    Array<Tx> *join(const int dim, const Array<Tx> &first, const Array<Ty> &second)
    {
        // All dimensions except join dimension must be equal
        // Compute output dims
        af::dim4 odims;
        af::dim4 fdims = first.dims();
        af::dim4 sdims = second.dims();

        for(int i = 0; i < 4; i++) {
            if(i == dim) {
                odims[i] = fdims[i] + sdims[i];
            } else {
                odims[i] = fdims[i];
            }
        }

        Array<Tx> *out = createEmptyArray<Tx>(odims);

        Tx* outPtr = out->get();
        const Tx* fptr = first.get();
        const Ty* sptr = second.get();

        switch(dim) {
            case 0: join_<Tx, Ty, 0>(outPtr, fptr, sptr, odims, fdims, sdims,
                                     out->strides(), first.strides(), second.strides());
                    break;
            case 1: join_<Tx, Ty, 1>(outPtr, fptr, sptr, odims, fdims, sdims,
                                     out->strides(), first.strides(), second.strides());
                    break;
            case 2: join_<Tx, Ty, 2>(outPtr, fptr, sptr, odims, fdims, sdims,
                                     out->strides(), first.strides(), second.strides());
                    break;
            case 3: join_<Tx, Ty, 3>(outPtr, fptr, sptr, odims, fdims, sdims,
                                     out->strides(), first.strides(), second.strides());
                    break;
        }

        return out;
    }

#define INSTANTIATE(Tx, Ty)                                                                             \
    template Array<Tx>* join<Tx, Ty>(const int dim, const Array<Tx> &first, const Array<Ty> &second);   \

    INSTANTIATE(float,   float)
    INSTANTIATE(double,  double)
    INSTANTIATE(cfloat,  cfloat)
    INSTANTIATE(cdouble, cdouble)
    INSTANTIATE(int,     int)
    INSTANTIATE(uint,    uint)
    INSTANTIATE(uchar,   uchar)
    INSTANTIATE(char,    char)
}
