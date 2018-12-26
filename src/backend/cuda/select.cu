/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <Array.hpp>
#include <common/jit/NaryNode.hpp>
#include <err_cuda.hpp>
#include <kernel/select.hpp>
#include <scalar.hpp>
#include <select.hpp>

using common::NaryNode;
using common::Node_ptr;

namespace cuda {
template<typename T>
void select(Array<T> &out, const Array<char> &cond, const Array<T> &a,
            const Array<T> &b) {
    kernel::select<T>(out, cond, a, b, out.ndims());
}

template<typename T, bool flip>
void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a,
                   const double &b) {
    kernel::select_scalar<T, flip>(out, cond, a, b, out.ndims());
}

template<typename T>
Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a,
                          const Array<T> &b, const af::dim4 &odims) {
    auto cond_node = cond.getNode();
    auto a_node    = a.getNode();
    auto b_node    = b.getNode();
    int height     = std::max(a_node->getHeight(), b_node->getHeight());
    height         = std::max(height, cond_node->getHeight()) + 1;

    NaryNode *node =
        new NaryNode(getFullName<T>(), shortname<T>(true), "__select", 3,
                     {{cond_node, a_node, b_node}}, (int)af_select_t, height);

    Array<T> out = createNodeArray<T>(odims, Node_ptr(node));
    return out;
}

template<typename T, bool flip>
Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a,
                          const double &b_val, const af::dim4 &odims) {
    auto cond_node = cond.getNode();
    auto a_node    = a.getNode();
    Array<T> b     = createScalarNode<T>(odims, scalar<T>(b_val));
    auto b_node    = b.getNode();
    int height     = std::max(a_node->getHeight(), b_node->getHeight());
    height         = std::max(height, cond_node->getHeight()) + 1;

    NaryNode *node = new NaryNode(
        getFullName<T>(), shortname<T>(true),
        flip ? "__not_select" : "__select", 3, {{cond_node, a_node, b_node}},
        (int)(flip ? af_not_select_t : af_select_t), height);

    Array<T> out = createNodeArray<T>(odims, Node_ptr(node));
    return out;
}

#define INSTANTIATE(T)                                                        \
    template Array<T> createSelectNode<T>(                                    \
        const Array<char> &cond, const Array<T> &a, const Array<T> &b,        \
        const af::dim4 &odims);                                               \
    template Array<T> createSelectNode<T, true>(                              \
        const Array<char> &cond, const Array<T> &a, const double &b_val,      \
        const af::dim4 &odims);                                               \
    template Array<T> createSelectNode<T, false>(                             \
        const Array<char> &cond, const Array<T> &a, const double &b_val,      \
        const af::dim4 &odims);                                               \
    template void select<T>(Array<T> & out, const Array<char> &cond,          \
                            const Array<T> &a, const Array<T> &b);            \
    template void select_scalar<T, true>(Array<T> & out,                      \
                                         const Array<char> &cond,             \
                                         const Array<T> &a, const double &b); \
    template void select_scalar<T, false>(Array<T> & out,                     \
                                          const Array<char> &cond,            \
                                          const Array<T> &a, const double &b)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(cfloat);
INSTANTIATE(cdouble);
INSTANTIATE(int);
INSTANTIATE(uint);
INSTANTIATE(intl);
INSTANTIATE(uintl);
INSTANTIATE(char);
INSTANTIATE(uchar);
INSTANTIATE(short);
INSTANTIATE(ushort);
}  // namespace cuda
