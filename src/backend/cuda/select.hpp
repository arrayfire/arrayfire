/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once
#include <JIT/NaryNode.hpp>
#include <scalar.hpp>
#include <Array.hpp>

namespace cuda
{
    template<typename T>
    void select(Array<T> &out, const Array<char> &cond, const Array<T> &a, const Array<T> &b);

    template<typename T, bool flip>
    void select_scalar(Array<T> &out, const Array<char> &cond, const Array<T> &a, const double &b);

    template<typename T>
    Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a, const Array<T> &b, const af::dim4 &odims)
    {
        auto cond_node = cond.getNode();
        auto a_node = a.getNode();
        auto b_node = b.getNode();
        int height = std::max(a_node->getHeight(), b_node->getHeight());
        height = std::max(height, cond_node->getHeight()) + 1;

        JIT::NaryNode *node = new JIT::NaryNode(getFullName<T>(), shortname<T>(true),
                                                "__select", 3, {{cond_node, a_node, b_node}},
                                                (int)af_select_t, height);

        Array<T> out = createNodeArray<T>(odims, JIT::Node_ptr(node));
        return out;
    }

    template<typename T, bool flip>
    Array<T> createSelectNode(const Array<char> &cond, const Array<T> &a, const double &b_val, const af::dim4 &odims)
    {
        auto cond_node = cond.getNode();
        auto a_node = a.getNode();
        Array<T> b = createScalarNode<T>(odims, scalar<T>(b_val));
        auto b_node = b.getNode();
        int height = std::max(a_node->getHeight(), b_node->getHeight());
        height = std::max(height, cond_node->getHeight()) + 1;

        JIT::NaryNode *node = new JIT::NaryNode(getFullName<T>(), shortname<T>(true),
                                                flip ? "__not_select" : "__select",
                                                3, {{cond_node, a_node, b_node}},
                                                (int)(flip ? af_not_select_t : af_select_t),
                                                height);

        Array<T> out = createNodeArray<T>(odims, JIT::Node_ptr(node));
        return out;
    }
}
