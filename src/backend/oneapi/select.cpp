/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
// #include <kernel/select.hpp>
#include <select.hpp>

#include <Array.hpp>
#include <common/half.hpp>
#include <common/jit/NaryNode.hpp>
#include <err_oneapi.hpp>
#include <kernel/select.hpp>
#include <scalar.hpp>

#include <nonstd/span.hpp>
#include <memory>

using af::dim4;

using arrayfire::common::half;
using arrayfire::common::NaryNode;

using std::make_shared;
using std::max;

namespace arrayfire {
namespace oneapi {
template<typename T>
Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a,
                          const Array<T> &b, const dim4 &odims) {
    auto cond_node   = cond.getNode();
    auto a_node      = a.getNode();
    auto b_node      = b.getNode();
    auto a_height    = a_node->getHeight();
    auto b_height    = b_node->getHeight();
    auto cond_height = cond_node->getHeight();
    const int height = max(max(a_height, b_height), cond_height) + 1;

    auto node = make_shared<NaryNode>(NaryNode(
        static_cast<af::dtype>(af::dtype_traits<T>::af_type), "__select", 3,
        {{cond_node, a_node, b_node}}, af_select_t, height));
    std::array<common::Node *, 1> nodes{node.get()};
    if (detail::passesJitHeuristics<T>(nodes) != kJITHeuristics::Pass) {
        if (a_height > max(b_height, cond_height)) {
            a.eval();
        } else if (b_height > cond_height) {
            b.eval();
        } else {
            cond.eval();
        }
        return createSelectNode<T>(cond, a, b, odims);
    }
    return createNodeArray<T>(odims, node);
}

template<typename T, bool flip>
Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a,
                          const T &b_val, const dim4 &odims) {
    auto cond_node   = cond.getNode();
    auto a_node      = a.getNode();
    Array<T> b       = createScalarNode<T>(odims, b_val);
    auto b_node      = b.getNode();
    auto a_height    = a_node->getHeight();
    auto b_height    = b_node->getHeight();
    auto cond_height = cond_node->getHeight();
    const int height = max(max(a_height, b_height), cond_height) + 1;

    auto node = make_shared<NaryNode>(NaryNode(
        static_cast<af::dtype>(af::dtype_traits<T>::af_type),
        (flip ? "__not_select" : "__select"), 3, {{cond_node, a_node, b_node}},
        (flip ? af_not_select_t : af_select_t), height));

    std::array<common::Node *, 1> nodes{node.get()};
    if (detail::passesJitHeuristics<T>(nodes) != kJITHeuristics::Pass) {
        if (a_height > max(b_height, cond_height)) {
            a.eval();
        } else if (b_height > cond_height) {
            b.eval();
        } else {
            cond.eval();
        }
        return createSelectNode<T, flip>(cond, a, b_val, odims);
    }
    return createNodeArray<T>(odims, node);
}

template<typename T>
void select(Array<T> &out, const Array<char> &cond, const Array<T> &a,
            const Array<T> &b) {
    if constexpr (!(std::is_same_v<T, double> || std::is_same_v<T, cdouble>)) {
        kernel::select<T>(out, cond, a, b, out.ndims());
    }
}

template<typename T, bool flip>
void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a,
                   const T &b) {
    if constexpr (!(std::is_same_v<T, double> || std::is_same_v<T, cdouble>)) {
        kernel::select_scalar<T>(out, cond, a, b, out.ndims(), flip);
    }
}

#define INSTANTIATE(T)                                                   \
    template Array<T> createSelectNode<T>(                               \
        const Array<char> &cond, const Array<T> &a, const Array<T> &b,   \
        const af::dim4 &odims);                                          \
    template Array<T> createSelectNode<T, true>(                         \
        const Array<char> &cond, const Array<T> &a, const T &b_val,      \
        const af::dim4 &odims);                                          \
    template Array<T> createSelectNode<T, false>(                        \
        const Array<char> &cond, const Array<T> &a, const T &b_val,      \
        const af::dim4 &odims);                                          \
    template void select<T>(Array<T> & out, const Array<char> &cond,     \
                            const Array<T> &a, const Array<T> &b);       \
    template void select_scalar<T, true>(Array<T> & out,                 \
                                         const Array<char> &cond,        \
                                         const Array<T> &a, const T &b); \
    template void select_scalar<T, false>(Array<T> & out,                \
                                          const Array<char> &cond,       \
                                          const Array<T> &a, const T &b)

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
INSTANTIATE(half);

#undef INSTANTIATE
}  // namespace oneapi
}  // namespace arrayfire
