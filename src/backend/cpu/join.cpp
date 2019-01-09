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
#include <platform.hpp>
#include <queue.hpp>

namespace cpu {

template <typename Tx, typename Ty>
Array<Tx> join(const int dim, const Array<Tx> &first, const Array<Ty> &second) {
    first.eval();
    second.eval();

    // All dimensions except join dimension must be equal
    // Compute output dims
    af::dim4 odims;
    af::dim4 fdims = first.dims();
    af::dim4 sdims = second.dims();

    for (int i = 0; i < 4; i++) {
        if (i == dim) {
            odims[i] = fdims[i] + sdims[i];
        } else {
            odims[i] = fdims[i];
        }
    }

    Array<Tx> out = createEmptyArray<Tx>(odims);

    getQueue().enqueue(kernel::join<Tx, Ty>, out, dim, first, second);

    return out;
}

template <typename T>
Array<T> join(const int dim, const std::vector<Array<T>> &inputs) {
    for (unsigned i = 0; i < inputs.size(); ++i) inputs[i].eval();
    // All dimensions except join dimension must be equal
    // Compute output dims
    af::dim4 odims;
    const dim_t n_arrays = inputs.size();
    std::vector<af::dim4> idims(n_arrays);

    dim_t dim_size = 0;
    for (unsigned i = 0; i < idims.size(); i++) {
        idims[i] = inputs[i].dims();
        dim_size += idims[i][dim];
    }

    for (int i = 0; i < 4; i++) {
        if (i == dim) {
            odims[i] = dim_size;
        } else {
            odims[i] = idims[0][i];
        }
    }

    std::vector<CParam<T>> inputParams(inputs.begin(), inputs.end());
    Array<T> out = createEmptyArray<T>(odims);

    switch (n_arrays) {
        case 1:
            getQueue().enqueue(kernel::join<T, 1>, dim, out, inputParams);
            break;
        case 2:
            getQueue().enqueue(kernel::join<T, 2>, dim, out, inputParams);
            break;
        case 3:
            getQueue().enqueue(kernel::join<T, 3>, dim, out, inputParams);
            break;
        case 4:
            getQueue().enqueue(kernel::join<T, 4>, dim, out, inputParams);
            break;
        case 5:
            getQueue().enqueue(kernel::join<T, 5>, dim, out, inputParams);
            break;
        case 6:
            getQueue().enqueue(kernel::join<T, 6>, dim, out, inputParams);
            break;
        case 7:
            getQueue().enqueue(kernel::join<T, 7>, dim, out, inputParams);
            break;
        case 8:
            getQueue().enqueue(kernel::join<T, 8>, dim, out, inputParams);
            break;
        case 9:
            getQueue().enqueue(kernel::join<T, 9>, dim, out, inputParams);
            break;
        case 10:
            getQueue().enqueue(kernel::join<T, 10>, dim, out, inputParams);
            break;
    }

    return out;
}

#define INSTANTIATE(Tx, Ty)                                                \
    template Array<Tx> join<Tx, Ty>(const int dim, const Array<Tx> &first, \
                                    const Array<Ty> &second);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(cdouble, cdouble)
INSTANTIATE(int, int)
INSTANTIATE(uint, uint)
INSTANTIATE(intl, intl)
INSTANTIATE(uintl, uintl)
INSTANTIATE(uchar, uchar)
INSTANTIATE(char, char)
INSTANTIATE(ushort, ushort)
INSTANTIATE(short, short)

#undef INSTANTIATE

#define INSTANTIATE(T)                       \
    template Array<T> join<T>(const int dim, \
                              const std::vector<Array<T>> &inputs);

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
INSTANTIATE(ushort)
INSTANTIATE(short)

#undef INSTANTIATE
}  // namespace cpu
