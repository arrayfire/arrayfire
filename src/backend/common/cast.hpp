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
#include <common/Logger.hpp>

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
    static spdlog::logger *getLogger() noexcept {
        static std::shared_ptr<spdlog::logger> logger =
            common::loggerFactory("ast");
        return logger.get();
    }
    detail::Array<To> operator()(const detail::Array<Ti> &in) {
        detail::CastOp<To, Ti> cop;
        common::Node_ptr in_node = in.getNode();
        af::dtype to_dtype = static_cast<af::dtype>(dtype_traits<To>::af_type);

        // JIT optimization in the cast of multiple sequential casts that become
        // idempotent - check to see if the previous operation was also a cast
        // TODO: handle arbitrarily long chains of casts
        auto in_node_unary =
            std::dynamic_pointer_cast<common::UnaryNode>(in_node);
        if (in_node_unary && in_node_unary->getOp() == af_cast_t) {
            // child child's output type is the input type of the child
            AF_TRACE("Cast optimiztion performed by removing cast to {}",
                     dtype_traits<Ti>::getName());
            auto in_in_node = in_node_unary->getChildren()[0];
            if (in_in_node->getType() == to_dtype) {
                // ignore the input node and simply connect a noop node from the
                // child's child to produce this op's output
                return detail::createNodeArray<To>(in.dims(), in_in_node);
            }
        }

        common::UnaryNode *node =
            new common::UnaryNode(to_dtype, cop.name(), in_node, af_cast_t);
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
