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
        for(dim_t ow = 0; ow < xdims[3]; ow++) {
            const dim_t xW = ow * xst[3];
            const dim_t oW = (ow + offset[3]) * ost[3];

            for(dim_t oz = 0; oz < xdims[2]; oz++) {
                const dim_t xZW = xW + oz * xst[2];
                const dim_t oZW = oW + (oz + offset[2]) * ost[2];

                for(dim_t oy = 0; oy < xdims[1]; oy++) {
                    const dim_t xYZW = xZW + oy * xst[1];
                    const dim_t oYZW = oZW + (oy + offset[1]) * ost[1];

                    for(dim_t ox = 0; ox < xdims[0]; ox++) {
                        const dim_t iMem = xYZW + ox;
                        const dim_t oMem = oYZW + (ox + offset[0]);
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

    template<typename T, int n_arrays>
    void join_wrapper(const int dim, Array<T> &out, const std::vector<Array<T>> &inputs)
    {
        af::dim4 zero(0,0,0,0);
        af::dim4 d = zero;
        switch(dim) {
            case 0:
                join_append<T, T, 0>(out.get(), inputs[0].get(), zero,
                            out.dims(), inputs[0].dims(), out.strides(), inputs[0].strides());
                for(int i = 1; i < n_arrays; i++) {
                    d += inputs[i - 1].dims();
                    join_append<T, T, 0>(out.get(), inputs[i].get(), calcOffset<0>(d),
                            out.dims(), inputs[i].dims(), out.strides(), inputs[i].strides());
                }
                break;
            case 1:
                join_append<T, T, 1>(out.get(), inputs[0].get(), zero,
                            out.dims(), inputs[0].dims(), out.strides(), inputs[0].strides());
                for(int i = 1; i < n_arrays; i++) {
                    d += inputs[i - 1].dims();
                    join_append<T, T, 1>(out.get(), inputs[i].get(), calcOffset<1>(d),
                            out.dims(), inputs[i].dims(), out.strides(), inputs[i].strides());
                }
                break;
            case 2:
                join_append<T, T, 2>(out.get(), inputs[0].get(), zero,
                            out.dims(), inputs[0].dims(), out.strides(), inputs[0].strides());
                for(int i = 1; i < n_arrays; i++) {
                    d += inputs[i - 1].dims();
                    join_append<T, T, 2>(out.get(), inputs[i].get(), calcOffset<2>(d),
                            out.dims(), inputs[i].dims(), out.strides(), inputs[i].strides());
                }
                break;
            case 3:
                join_append<T, T, 3>(out.get(), inputs[0].get(), zero,
                            out.dims(), inputs[0].dims(), out.strides(), inputs[0].strides());
                for(int i = 1; i < n_arrays; i++) {
                    d += inputs[i - 1].dims();
                    join_append<T, T, 3>(out.get(), inputs[i].get(), calcOffset<3>(d),
                            out.dims(), inputs[i].dims(), out.strides(), inputs[i].strides());
                }
                break;
        }
    }

    template<typename T>
    Array<T> join(const int dim, const std::vector<Array<T>> &inputs)
    {
        // All dimensions except join dimension must be equal
        // Compute output dims
        af::dim4 odims;
        const dim_t n_arrays = inputs.size();
        std::vector<af::dim4> idims(n_arrays);

        dim_t dim_size = 0;
        for(int i = 0; i < (int)idims.size(); i++) {
            idims[i] = inputs[i].dims();
            dim_size += idims[i][dim];
        }

        for(int i = 0; i < 4; i++) {
            if(i == dim) {
                odims[i] = dim_size;
            } else {
                odims[i] = idims[0][i];
            }
        }

        Array<T> out = createEmptyArray<T>(odims);

        switch(n_arrays) {
            case 1:
                join_wrapper<T, 1>(dim, out, inputs);
                break;
            case 2:
                join_wrapper<T, 2>(dim, out, inputs);
                break;
            case 3:
                join_wrapper<T, 3>(dim, out, inputs);
                break;
            case 4:
                join_wrapper<T, 4>(dim, out, inputs);
                break;
            case 5:
                join_wrapper<T, 5>(dim, out, inputs);
                break;
            case 6:
                join_wrapper<T, 6>(dim, out, inputs);
                break;
            case 7:
                join_wrapper<T, 7>(dim, out, inputs);
                break;
            case 8:
                join_wrapper<T, 8>(dim, out, inputs);
                break;
            case 9:
                join_wrapper<T, 9>(dim, out, inputs);
                break;
            case 10:
                join_wrapper<T,10>(dim, out, inputs);
                break;
        }

        return out;
    }

#define INSTANTIATE(Tx, Ty) \
    template Array<Tx> join<Tx, Ty>(const int dim, const Array<Tx> &first, const Array<Ty> &second);

    INSTANTIATE(float,   float)
    INSTANTIATE(double,  double)
    INSTANTIATE(cfloat,  cfloat)
    INSTANTIATE(cdouble, cdouble)
    INSTANTIATE(int,     int)
    INSTANTIATE(uint,    uint)
    INSTANTIATE(intl,    intl)
    INSTANTIATE(uintl,   uintl)
    INSTANTIATE(uchar,   uchar)
    INSTANTIATE(char,    char)

#undef INSTANTIATE

#define INSTANTIATE(T)      \
    template Array<T> join<T>(const int dim, const std::vector<Array<T>> &inputs);

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
    INSTANTIATE(uchar)
    INSTANTIATE(char)

#undef INSTANTIATE
}
