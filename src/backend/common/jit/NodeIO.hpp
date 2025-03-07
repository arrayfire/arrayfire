/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/jit/Node.hpp>
#include <spdlog/fmt/bundled/format.h>

#include <common/util.hpp>

template<>
struct fmt::formatter<af::dtype> : fmt::formatter<char> {
    template<typename FormatContext>
    auto format(const af::dtype& p, FormatContext& ctx) -> decltype(ctx.out()) {
        format_to(ctx.out(), "{}", arrayfire::common::getName(p));
        return ctx.out();
    }
};

template<>
struct fmt::formatter<arrayfire::common::Node> {
    // Presentation format: 'p' - pointer, 't' - type.
    // char presentation;
    bool pointer;
    bool type;
    bool children;
    bool op;

    // Parses format specifications of the form ['f' | 'e'].
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();

        if (it == end || *it == '}') {
            pointer = type = children = op = true;
            return it;
        }

        while (it != end && *it != '}') {
            switch (*it) {
                case 'p': pointer = true; break;
                case 't': type = true; break;
                case 'c': children = true; break;
                case 'o': op = true; break;
                default: throw format_error("invalid format");
            }
            ++it;
        }

        // Return an iterator past the end of the parsed range:
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    template<typename FormatContext>
    auto format(const arrayfire::common::Node& node, FormatContext& ctx)
        -> decltype(ctx.out()) {
        // ctx.out() is an output iterator to write to.

        format_to(ctx.out(), "{{");
        if (pointer) format_to(ctx.out(), "{} ", (void*)&node);
        if (op) {
            if (isBuffer(node)) {
                format_to(ctx.out(), "buffer ");
            } else if (isScalar(node)) {
                format_to(ctx.out(), "scalar ",
                          arrayfire::common::toString(node.getOp()));
            } else {
                format_to(ctx.out(), "{} ",
                          arrayfire::common::toString(node.getOp()));
            }
        }
        if (type) format_to(ctx.out(), "{} ", node.getType());
        if (children) {
            int count;
            for (count = 0; count < arrayfire::common::Node::kMaxChildren &&
                            node.m_children[count].get() != nullptr;
                 count++) {}
            if (count > 0) {
                format_to(ctx.out(), "children: {{ ");
                for (int i = 0; i < count; i++) {
                    format_to(ctx.out(), "{} ", *(node.m_children[i].get()));
                }
                format_to(ctx.out(), "\b}} ");
            }
        }
        format_to(ctx.out(), "\b}}");

        return ctx.out();
    }
};
