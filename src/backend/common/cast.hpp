/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Array.hpp>
#include <cast.hpp>

#ifdef AF_CPU
#include <jit/UnaryNode.hpp>
#endif

namespace common {

#ifdef AF_CPU
template<typename To, typename Ti>
struct CastWrapper {
    detail::Array<To> operator()(const detail::Array<Ti> &in) {
        using cpu::jit::UnaryNode;

        Node_ptr in_node = in.getNode();
        auto node = std::make_shared<UnaryNode<To, Ti, af_cast_t>>(in_node);

        return detail::createNodeArray<To>(in.dims(), move(node));
    }
};
#else
template<typename To, typename Ti>
struct CastWrapper {
    detail::Array<To> operator()(const detail::Array<Ti> &in) {
        detail::CastOp<To, Ti> cop;
        common::Node_ptr in_node = in.getNode();
        common::UnaryNode *node  = new common::UnaryNode(
            static_cast<af::dtype>(dtype_traits<To>::af_type), cop.name(),
            in_node, af_cast_t);
        return detail::createNodeArray<To>(in.dims(), common::Node_ptr(node));
    }
};
#endif

template<typename T>
struct CastWrapper<T, T> {
    detail::Array<T> operator()(const detail::Array<T> &in);
};

template<typename To, typename Ti>
auto cast(detail::Array<Ti> &&in)
    -> std::enable_if_t<std::is_same<Ti, To>::value, detail::Array<To>> {
    return std::move(in);
}

template<typename To, typename Ti>
auto cast(const detail::Array<Ti> &in)
    -> std::enable_if_t<std::is_same<Ti, To>::value, detail::Array<To>> {
    return in;
}

template<typename To, typename Ti>
auto cast(const detail::Array<Ti> &in)
    -> std::enable_if_t<std::is_same<Ti, To>::value == false,
                        detail::Array<To>> {
    CastWrapper<To, Ti> cast_op;
    return cast_op(in);
}

}  // namespace common
