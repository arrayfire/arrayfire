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
#include <kernel/join.hpp>
#include <stdexcept>
#include <err_cuda.hpp>

namespace cuda
{
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

        af::dim4 zero(0,0,0,0);

        switch(dim) {
            case 0:
                kernel::join<Tx, Tx, 0>(out, first,  zero);
                kernel::join<Tx, Ty, 0>(out, second, calcOffset<0>(fdims));
                break;
            case 1:
                kernel::join<Tx, Tx, 1>(out, first,  zero);
                kernel::join<Tx, Ty, 1>(out, second, calcOffset<1>(fdims));
                break;
            case 2:
                kernel::join<Tx, Tx, 2>(out, first,  zero);
                kernel::join<Tx, Ty, 2>(out, second, calcOffset<2>(fdims));
                break;
            case 3:
                kernel::join<Tx, Tx, 3>(out, first,  zero);
                kernel::join<Tx, Ty, 3>(out, second, calcOffset<3>(fdims));
                break;
        }

        return out;
    }

    template<typename T, int n_arrays>
    void join_wrapper(const int dim, Array<T> &out, const std::vector<Array<T> > &inputs)
    {
        af::dim4 zero(0,0,0,0);
        af::dim4 d = zero;

        switch(dim) {
            case 0:
                kernel::join<T, T, 0>(out, inputs[0], zero);
                for(int i = 1; i < n_arrays; i++) {
                    d += inputs[i - 1].dims();
                    kernel::join<T, T, 0>(out, inputs[i], calcOffset<0>(d));
                }
                break;
            case 1:
                kernel::join<T, T, 1>(out, inputs[0], zero);
                for(int i = 1; i < n_arrays; i++) {
                    d += inputs[i - 1].dims();
                    kernel::join<T, T, 1>(out, inputs[i], calcOffset<1>(d));
                }
                break;
            case 2:
                kernel::join<T, T, 1>(out, inputs[0], zero);
                for(int i = 1; i < n_arrays; i++) {
                    d += inputs[i - 1].dims();
                    kernel::join<T, T, 2>(out, inputs[i], calcOffset<2>(d));
                }
                break;
            case 3:
                kernel::join<T, T, 3>(out, inputs[0], zero);
                for(int i = 1; i < n_arrays; i++) {
                    d += inputs[i - 1].dims();
                    kernel::join<T, T, 3>(out, inputs[i], calcOffset<3>(d));
                }
                break;
        }
    }

    template<typename T>
    Array<T> join(const int dim, const std::vector<Array<T> > &inputs)
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

#define INSTANTIATE(Tx, Ty)                                                                             \
    template Array<Tx> join<Tx, Ty>(const int dim, const Array<Tx> &first, const Array<Ty> &second);   \

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

#define INSTANTIATE(T)                                                                              \
    template Array<T> join<T>(const int dim, const std::vector<Array<T> > &inputs);

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
