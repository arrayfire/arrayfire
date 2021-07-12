/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <err_cuda.hpp>
#include <join.hpp>
#include <kernel/memcopy.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

using af::dim4;
using common::half;
using std::vector;

namespace cuda {

template<typename T>
Array<T> join(const int jdim, const Array<T> &first, const Array<T> &second) {
    // All dimensions except join dimension must be equal
    const dim4 fdims = first.dims();
    const dim4 sdims = second.dims();
    // Compute output dims
    dim4 odims(fdims);
    odims.dims[jdim] += sdims.dims[jdim];

    Array<T> out = createEmptyArray<T>(odims);
    vector<kernel::CParamPlus<T>> ins;
    ins.reserve(2);

    if (first.isReady()) {
        ins.emplace_back((CParam<T>)first, 0);
    } else {
        vector<Param<T>> outputs{{out.get(), fdims.dims, out.strides().dims}};
        const vector<common::Node *> nodes{first.getNode().get()};
        evalNodes<T>(outputs, nodes);
    }

    if (second.isReady()) {
        ins.emplace_back((CParam<T>)second,
                         fdims.dims[jdim] * out.strides().dims[jdim]);
    } else {
        vector<Param<T>> outputs{
            {out.get() + fdims.dims[jdim] * out.strides().dims[jdim],
             sdims.dims, out.strides().dims}};
        const vector<common::Node *> nodes{second.getNode().get()};
        evalNodes<T>(outputs, nodes);
    }

    if (ins.size()) kernel::memcopyN<T>(out, ins);
    return out;
}

template<typename T>
Array<T> join(const int jdim, const std::vector<Array<T>> &inputs) {
    // All dimensions except join dimension must be equal
    // Compute output dims
    dim4 odims(inputs[0].dims());
    odims.dims[jdim] = 0;
    for (auto &iArray : inputs) {
        odims.dims[jdim] += iArray.dims().dims[jdim];
    }

    Array<T> out = createEmptyArray<T>(odims);
    vector<kernel::CParamPlus<T>> ins;
    ins.reserve(inputs.size());

    dim_t outOffset = 0;
    for (auto &iArray : inputs) {
        if (iArray.isReady()) {
            ins.emplace_back((CParam<T>)iArray, outOffset);
        } else {
            vector<Param<T>> outputs{{out.get() + outOffset, iArray.dims().dims,
                                      out.strides().dims}};
            vector<common::Node *> nodes{iArray.getNode().get()};
            evalNodes<T>(outputs, nodes);
        }
        outOffset += iArray.dims().dims[jdim] * out.strides().dims[jdim];
    }

    if (ins.size()) kernel::memcopyN<T>(out, ins);

    return out;
}

#define INSTANTIATE(T)                                               \
    template Array<T> join<T>(const int jdim, const Array<T> &first, \
                              const Array<T> &second);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(half)

#undef INSTANTIATE

#define INSTANTIATE(T)                        \
    template Array<T> join<T>(const int jdim, \
                              const std::vector<Array<T>> &inputs);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(half)

#undef INSTANTIATE
}  // namespace cuda
