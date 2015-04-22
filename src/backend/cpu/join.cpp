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
    template<typename To, typename Tx, int dim>
    void join_append(To *out, const Tx *X, const af::dim4 &offset,
               const af::dim4 &odims, const af::dim4 &xdims,
               const af::dim4 &ost, const af::dim4 &xst)
    {
        for(dim_type ow = 0; ow < xdims[3]; ow++) {
            const dim_type xW = ow * xst[3];
            const dim_type oW = (ow + offset[3]) * ost[3];

            for(dim_type oz = 0; oz < xdims[2]; oz++) {
                const dim_type xZW = xW + oz * xst[2];
                const dim_type oZW = oW + (oz + offset[2]) * ost[2];

                for(dim_type oy = 0; oy < xdims[1]; oy++) {
                    const dim_type xYZW = xZW + oy * xst[1];
                    const dim_type oYZW = oZW + (oy + offset[1]) * ost[1];

                    for(dim_type ox = 0; ox < xdims[0]; ox++) {
                        const dim_type iMem = xYZW + ox;
                        const dim_type oMem = oYZW + (ox + offset[0]);
                        out[oMem] = X[iMem];
                    }
                }
            }
        }
    }

    template<int dim>
    af::dim4 calcOffset(const af::dim4 dims)
    {
        af::dim4 offset;
        offset[0] = (dim == 0) ? dims[0] : 0;
        offset[1] = (dim == 1) ? dims[1] : 0;
        offset[2] = (dim == 2) ? dims[2] : 0;
        offset[3] = (dim == 3) ? dims[3] : 0;
        return offset;
    }

    template<typename Tx, typename Ty>
    Array<Tx> join(const int dim, const Array<Tx> &first, const Array<Ty> &second)
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

        Array<Tx> out = createEmptyArray<Tx>(odims);

        Tx* outPtr = out.get();
        const Tx* fptr = first.get();
        const Ty* sptr = second.get();

        af::dim4 zero(0,0,0,0);

        switch(dim) {
            case 0:
                join_append<Tx, Tx, 0>(outPtr, fptr, zero,
                                       odims, fdims, out.strides(), first.strides());
                join_append<Tx, Ty, 0>(outPtr, sptr, calcOffset<0>(fdims),
                                       odims, sdims, out.strides(), second.strides());
                break;
            case 1:
                join_append<Tx, Tx, 1>(outPtr, fptr, zero,
                                       odims, fdims, out.strides(), first.strides());
                join_append<Tx, Ty, 1>(outPtr, sptr, calcOffset<1>(fdims),
                                       odims, sdims, out.strides(), second.strides());
                break;
            case 2:
                join_append<Tx, Tx, 2>(outPtr, fptr, zero,
                                       odims, fdims, out.strides(), first.strides());
                join_append<Tx, Ty, 2>(outPtr, sptr, calcOffset<2>(fdims),
                                       odims, sdims, out.strides(), second.strides());
                break;
            case 3:
                join_append<Tx, Tx, 3>(outPtr, fptr, zero,
                                       odims, fdims, out.strides(), first.strides());
                join_append<Tx, Ty, 3>(outPtr, sptr, calcOffset<3>(fdims),
                                       odims, sdims, out.strides(), second.strides());
                break;
        }

        return out;
    }

    template<typename T>
    Array<T> join(const int dim, const Array<T> &first, const Array<T> &second,
                  const Array<T> &third)
    {
        // All dimensions except join dimension must be equal
        // Compute output dims
        af::dim4 odims;
        af::dim4 fdims = first.dims();
        af::dim4 sdims = second.dims();
        af::dim4 tdims = third.dims();

        for(int i = 0; i < 4; i++) {
            if(i == dim) {
                odims[i] = fdims[i] + sdims[i] + tdims[i];
            } else {
                odims[i] = fdims[i];
            }
        }

        Array<T> out = createEmptyArray<T>(odims);

        T* outPtr = out.get();
        const T* fptr = first.get();
        const T* sptr = second.get();
        const T* tptr = third.get();

        af::dim4 zero(0,0,0,0);

        switch(dim) {
            case 0:
                join_append<T, T, 0>(outPtr, fptr, zero,
                                     odims, fdims, out.strides(), first.strides());
                join_append<T, T, 0>(outPtr, sptr, calcOffset<0>(fdims),
                                     odims, sdims, out.strides(), second.strides());
                join_append<T, T, 0>(outPtr, tptr, calcOffset<0>(fdims + sdims),
                                     odims, tdims, out.strides(), third.strides());
                break;
            case 1:
                join_append<T, T, 1>(outPtr, fptr, zero,
                                     odims, fdims, out.strides(), first.strides());
                join_append<T, T, 1>(outPtr, sptr, calcOffset<1>(fdims),
                                     odims, sdims, out.strides(), second.strides());
                join_append<T, T, 1>(outPtr, tptr, calcOffset<1>(fdims + sdims),
                                     odims, tdims, out.strides(), third.strides());
                break;
            case 2:
                join_append<T, T, 2>(outPtr, fptr, zero,
                                     odims, fdims, out.strides(), first.strides());
                join_append<T, T, 2>(outPtr, sptr, calcOffset<2>(fdims),
                                     odims, sdims, out.strides(), second.strides());
                join_append<T, T, 2>(outPtr, tptr, calcOffset<2>(fdims + sdims),
                                     odims, tdims, out.strides(), third.strides());
                break;
            case 3:
                join_append<T, T, 3>(outPtr, fptr, zero,
                                     odims, fdims, out.strides(), first.strides());
                join_append<T, T, 3>(outPtr, sptr, calcOffset<3>(fdims),
                                     odims, sdims, out.strides(), second.strides());
                join_append<T, T, 3>(outPtr, tptr, calcOffset<3>(fdims + sdims),
                                       odims, tdims, out.strides(), third.strides());
                break;
        }

        return out;
    }

    template<typename T>
    Array<T> join(const int dim, const Array<T> &first, const Array<T> &second,
                  const Array<T> &third, const Array<T> &fourth)
    {
        // All dimensions except join dimension must be equal
        // Compute output dims
        af::dim4 odims;
        af::dim4 fdims = first.dims();
        af::dim4 sdims = second.dims();
        af::dim4 tdims = third.dims();
        af::dim4 rdims = fourth.dims();

        for(int i = 0; i < 4; i++) {
            if(i == dim) {
                odims[i] = fdims[i] + sdims[i] + tdims[i] + rdims[i];
            } else {
                odims[i] = fdims[i];
            }
        }

        Array<T> out = createEmptyArray<T>(odims);

        T* outPtr = out.get();
        const T* fptr = first.get();
        const T* sptr = second.get();
        const T* tptr = third.get();
        const T* rptr = fourth.get();

        af::dim4 zero(0,0,0,0);

        switch(dim) {
            case 0:
                join_append<T, T, 0>(outPtr, fptr, zero,
                                     odims, fdims, out.strides(), first.strides());
                join_append<T, T, 0>(outPtr, sptr, calcOffset<0>(fdims),
                                     odims, sdims, out.strides(), second.strides());
                join_append<T, T, 0>(outPtr, tptr, calcOffset<0>(fdims + sdims),
                                     odims, tdims, out.strides(), third.strides());
                join_append<T, T, 0>(outPtr, rptr, calcOffset<0>(fdims + sdims + tdims),
                                     odims, rdims, out.strides(), fourth.strides());
                break;
            case 1:
                join_append<T, T, 1>(outPtr, fptr, zero,
                                     odims, fdims, out.strides(), first.strides());
                join_append<T, T, 1>(outPtr, sptr, calcOffset<1>(fdims),
                                     odims, sdims, out.strides(), second.strides());
                join_append<T, T, 1>(outPtr, tptr, calcOffset<1>(fdims + sdims),
                                     odims, tdims, out.strides(), third.strides());
                join_append<T, T, 1>(outPtr, rptr, calcOffset<1>(fdims + sdims + tdims),
                                     odims, rdims, out.strides(), fourth.strides());
                break;
            case 2:
                join_append<T, T, 2>(outPtr, fptr, zero,
                                     odims, fdims, out.strides(), first.strides());
                join_append<T, T, 2>(outPtr, sptr, calcOffset<2>(fdims),
                                     odims, sdims, out.strides(), second.strides());
                join_append<T, T, 2>(outPtr, tptr, calcOffset<2>(fdims + sdims),
                                     odims, tdims, out.strides(), third.strides());
                join_append<T, T, 2>(outPtr, rptr, calcOffset<2>(fdims + sdims + tdims),
                                     odims, rdims, out.strides(), fourth.strides());
                break;
            case 3:
                join_append<T, T, 3>(outPtr, fptr, zero,
                                     odims, fdims, out.strides(), first.strides());
                join_append<T, T, 3>(outPtr, sptr, calcOffset<3>(fdims),
                                     odims, sdims, out.strides(), second.strides());
                join_append<T, T, 3>(outPtr, tptr, calcOffset<3>(fdims + sdims),
                                     odims, tdims, out.strides(), third.strides());
                join_append<T, T, 3>(outPtr, rptr, calcOffset<3>(fdims + sdims + tdims),
                                     odims, rdims, out.strides(), fourth.strides());
                break;
        }

        return out;
    }

#define INSTANTIATE(Tx, Ty)                                                                             \
    template Array<Tx> join<Tx, Ty>(const int dim, const Array<Tx> &first, const Array<Ty> &second);   \

    INSTANTIATE(float,   float)
    INSTANTIATE(double,  double)
    INSTANTIATE(cfloat,  cfloat)
    INSTANTIATE(cdouble, cdouble)
    INSTANTIATE(int,     int)
    INSTANTIATE(uint,    uint)
    INSTANTIATE(uchar,   uchar)
    INSTANTIATE(char,    char)

#undef INSTANTIATE

#define INSTANTIATE(T)                                                                              \
    template Array<T> join<T>(const int dim, const Array<T> &first, const Array<T> &second,         \
                              const Array<T> &third);                                               \
    template Array<T> join<T>(const int dim, const Array<T> &first, const Array<T> &second,         \
                              const Array<T> &third, const Array<T> &fourth);

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

#undef INSTANTIATE
}
