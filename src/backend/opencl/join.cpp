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
#include <err_opencl.hpp>
#include <join.hpp>
#include <kernel/memcopy.hpp>

#include <algorithm>
#include <stdexcept>
#include <vector>

using af::dim4;
using common::half;
using std::transform;
using std::vector;

namespace opencl {
template<typename T>
Array<T> join(const int jdim, const Array<T> &first, const Array<T> &second) {
    // All dimensions except join dimension must be equal
    const dim4 &fdims = first.dims();
    const dim4 &sdims = second.dims();
    // Compute output dims
    dim4 odims(fdims);
    odims.dims[jdim] += sdims.dims[jdim];

    Array<T> out = createEmptyArray<T>(odims);
    vector<kernel::BufferPlus> ins;
    ins.reserve(2);

    if (first.isReady()) {
        ins.emplace_back(first.getData().get(), fdims, first.strides(),
                         first.getOffset(), 0);
    } else {
        vector<Param> outputs{
            {out.get(),
             {{fdims.dims[0], fdims.dims[1], fdims.dims[2], fdims.dims[3]},
              {out.strides().dims[0], out.strides().dims[1],
               out.strides().dims[2], out.strides().dims[3]},
              0}}};
        const vector<common::Node *> nodes{first.getNode().get()};
        evalNodes(outputs, nodes);
    }

    if (second.isReady()) {
        ins.emplace_back(second.getData().get(), sdims, second.strides(),
                         second.getOffset(),
                         fdims.dims[jdim] * out.strides().dims[jdim]);
    } else {
        vector<Param> outputs{
            {out.get(),
             {{sdims.dims[0], sdims.dims[1], sdims.dims[2], sdims.dims[3]},
              {out.strides().dims[0], out.strides().dims[1],
               out.strides().dims[2], out.strides().dims[3]},
              fdims.dims[jdim] * out.strides().dims[jdim]}}};
        const vector<common::Node *> nodes{second.getNode().get()};
        evalNodes(outputs, nodes);
    }

    if (ins.size()) kernel::memcopyN<T>(*out.get(), out.strides(), ins);
    return out;
}

template<typename T>
Array<T> join(const int jdim, const vector<Array<T>> &inputs) {
    // All dimensions except join dimension must be equal
    // Compute output dims
    dim4 odims(inputs[0].dims());
    odims.dims[jdim] = 0;
    for (auto &iArray : inputs) odims.dims[jdim] += iArray.dims().dims[jdim];

    Array<T> out = createEmptyArray<T>(odims);
    vector<kernel::BufferPlus> ins;
    ins.reserve(inputs.size());

    dim_t outOffset      = out.getOffset();
    const dim4 &ostrides = out.strides();
    for (auto &iArray : inputs) {
        const dim4 &idims = iArray.dims();
        if (iArray.isReady()) {
            ins.emplace_back(iArray.get(), idims, iArray.strides(),
                             iArray.getOffset(), outOffset);
        } else {
            vector<Param> outputs{
                {out.get(),
                 {{idims.dims[0], idims.dims[1], idims.dims[2], idims.dims[3]},
                  {ostrides.dims[0], ostrides.dims[1], ostrides.dims[2],
                   ostrides.dims[3]},
                  outOffset}}};
            const vector<common::Node *> nodes{iArray.getNode().get()};
            evalNodes(outputs, nodes);
        }
        outOffset += idims.dims[jdim] * ostrides.dims[jdim];
    }

    if (ins.size()) kernel::memcopyN<T>(*out.get(), ostrides, ins);
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

#define INSTANTIATE(T) \
    template Array<T> join<T>(const int jdim, const vector<Array<T>> &inputs);

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
}  // namespace opencl
