/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/moddims.hpp>

#include <common/jit/ModdimNode.hpp>
#include <common/jit/NodeIterator.hpp>
#include <copy.hpp>

using af::dim4;
using detail::Array;
using detail::copyArray;
using detail::createNodeArray;

using std::make_shared;
using std::shared_ptr;
using std::array;
using arrayfire::common::Node;
using arrayfire::common::Node_ptr;
using std::vector;

namespace arrayfire {
namespace common {

Node_ptr copyModdims(const Node_ptr &in, const af::dim4 &newDim) {

    Node_ptr out = in->clone();
    for(int i = 0; i < in->kMaxChildren && in->m_children[i] != nullptr; ++i) {
        out->m_children[i] = copyModdims(in->m_children[i], newDim);
    }
    if(out->isBuffer()) out->modDims(newDim);

    return out;
}

template<typename T>
Array<T> moddimOp(const Array<T> &in, af::dim4 outDim) {

    const auto &node = in.getNode();

    NodeIterator<> it(node.get());

    dim4 olddims_t = in.dims();

    bool all_linear = true;
    while (all_linear && it != NodeIterator<>()) {
        all_linear &= it->isLinear(olddims_t.get());
        ++it;
    }
    if (all_linear == false) in.eval();

    Array<T> out = createNodeArray<T>(outDim, copyModdims(in.getNode(), outDim));

    return out;
}

template<typename T>
Array<T> modDims(const Array<T> &in, const af::dim4 &newDims) {
    if (in.isLinear() == false) {
        // Nonlinear array's shape cannot be modified. Copy the data and modify
        // the shape of the array
        Array<T> out = copyArray<T>(in);
        out.setDataDims(newDims);
        return out;
    } else if (in.isReady()) {
        /// If the array is a buffer, modify the dimension and return
        auto out = in;
        out.setDataDims(newDims);
        return out;
    } else {
        /// If the array is a node and not linear and not a buffer, then create
        /// a moddims node
        auto out = moddimOp<T>(in, newDims);
        return out;
    }
}

template<typename T>
detail::Array<T> flat(const detail::Array<T> &in) {
    const af::dim4 newDims(in.elements());
    return common::modDims<T>(in, newDims);
}

}  // namespace common
}  // namespace arrayfire

#define INSTANTIATE(TYPE)                                          \
    template detail::Array<TYPE> arrayfire::common::modDims<TYPE>( \
        const detail::Array<TYPE> &in, const af::dim4 &newDims);   \
    template detail::Array<TYPE> arrayfire::common::flat<TYPE>(    \
        const detail::Array<TYPE> &in)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(detail::cfloat);
INSTANTIATE(detail::cdouble);
INSTANTIATE(arrayfire::common::half);
INSTANTIATE(signed char);
INSTANTIATE(unsigned char);
INSTANTIATE(char);
INSTANTIATE(unsigned short);
INSTANTIATE(short);
INSTANTIATE(unsigned);
INSTANTIATE(int);
INSTANTIATE(long long);
INSTANTIATE(unsigned long long);
