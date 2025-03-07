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
#include <memory>

#ifdef AF_CPU
#include <jit/UnaryNode.hpp>
#endif

namespace arrayfire {
namespace common {
/// This function determines if consecutive cast operations should be
/// removed from a JIT AST.
///
/// This function returns true if consecutive cast operations in the JIT AST
/// should be removed. Multiple cast operations are removed when going from
/// a smaller type to a larger type and back again OR if the conversion is
/// between two floating point types including complex types.
///
///                  Cast operations that will be removed
///                        outer -> inner -> outer
///
///                                inner cast
///           f32  f64  c32  c64  s32  u32   u8   b8  s64  u64  s16  u16  f16
///     f32    x    x    x    x                                            x
///     f64    x    x    x    x                                            x
///  o  c32    x    x    x    x                                            x
///  u  c64    x    x    x    x                                            x
///  t  s32    x    x    x    x    x    x              x    x              x
///  e  u32    x    x    x    x    x    x              x    x              x
///  r   u8    x    x    x    x    x    x    x    x    x    x    x    x    x
///      b8    x    x    x    x    x    x    x    x    x    x    x    x    x
///  c  s64    x    x    x    x                        x    x              x
///  a  u64    x    x    x    x                        x    x              x
///  s  s16    x    x    x    x    x    x              x    x    x    x    x
///  t  u16    x    x    x    x    x    x              x    x    x    x    x
///     f16    x    x    x    x                                            x
///
/// \param[in] outer The type of the second cast and the child of the
///            previous cast
/// \param[in] inner  The type of the first cast
///
/// \returns True if the inner cast operation should be removed
constexpr bool canOptimizeCast(af::dtype outer, af::dtype inner) {
    if (isFloating(outer)) {
        if (isFloating(inner)) { return true; }
    } else {
        if (isFloating(inner)) { return true; }
        if (dtypeSize(inner) >= dtypeSize(outer)) { return true; }
    }

    return false;
}

#ifdef AF_CPU
template<typename To, typename Ti>
struct CastWrapper {
    static spdlog::logger *getLogger() noexcept {
        static std::shared_ptr<spdlog::logger> logger =
            common::loggerFactory("ast");
        return logger.get();
    }

    detail::Array<To> operator()(const detail::Array<Ti> &in) {
        using detail::jit::UnaryNode;

        common::Node_ptr in_node = in.getNode();
        constexpr af::dtype to_dtype =
            static_cast<af::dtype>(af::dtype_traits<To>::af_type);
        constexpr af::dtype in_dtype =
            static_cast<af::dtype>(af::dtype_traits<Ti>::af_type);

        if (canOptimizeCast(to_dtype, in_dtype)) {
            // JIT optimization in the cast of multiple sequential casts that
            // become idempotent - check to see if the previous operation was
            // also a cast
            // TODO: handle arbitrarily long chains of casts
            auto in_node_unary =
                std::dynamic_pointer_cast<UnaryNode<To, Ti, af_cast_t>>(
                    in_node);

            if (in_node_unary && in_node_unary->getOp() == af_cast_t) {
                // child child's output type is the input type of the child
                AF_TRACE("Cast optimiztion performed by removing cast to {}",
                         af::dtype_traits<Ti>::getName());
                auto in_child_node = in_node_unary->getChildren()[0];
                if (in_child_node->getType() == to_dtype) {
                    // ignore the input node and simply connect a noop node from
                    // the child's child to produce this op's output
                    return detail::createNodeArray<To>(in.dims(),
                                                       in_child_node);
                }
            }
        }

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
        using arrayfire::common::UnaryNode;
        detail::CastOp<To, Ti> cop;
        common::Node_ptr in_node = in.getNode();
        constexpr af::dtype to_dtype =
            static_cast<af::dtype>(af::dtype_traits<To>::af_type);
        constexpr af::dtype in_dtype =
            static_cast<af::dtype>(af::dtype_traits<Ti>::af_type);

        if (canOptimizeCast(to_dtype, in_dtype)) {
            // JIT optimization in the cast of multiple sequential casts that
            // become idempotent - check to see if the previous operation was
            // also a cast
            // TODO: handle arbitrarily long chains of casts
            auto in_node_unary =
                std::dynamic_pointer_cast<common::UnaryNode>(in_node);

            if (in_node_unary && in_node_unary->getOp() == af_cast_t) {
                // child child's output type is the input type of the child
                AF_TRACE("Cast optimiztion performed by removing cast to {}",
                         af::dtype_traits<Ti>::getName());
                auto in_child_node = in_node_unary->getChildren()[0];
                if (in_child_node->getType() == to_dtype) {
                    // ignore the input node and simply connect a noop node from
                    // the child's child to produce this op's output
                    return detail::createNodeArray<To>(in.dims(),
                                                       in_child_node);
                }
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
}  // namespace arrayfire
