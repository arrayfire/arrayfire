/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <af/seq.h>

template<>
struct fmt::formatter<af_seq> {
    // Parses format specifications of the form ['f' | 'e'].
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    template<typename FormatContext>
    auto format(const af_seq& p, FormatContext& ctx) -> decltype(ctx.out()) {
        // ctx.out() is an output iterator to write to.
        if (p.begin == af_span.begin && p.end == af_span.end &&
            p.step == af_span.step) {
            return format_to(ctx.out(), "span");
        }
        if (p.begin == p.end) { return format_to(ctx.out(), "{}", p.begin); }
        if (p.step == 1) {
            return format_to(ctx.out(), "({} -> {})", p.begin, p.end);
        }
        return format_to(ctx.out(), "({} -({})-> {})", p.begin, p.step, p.end);
    }
};
