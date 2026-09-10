/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/Version.hpp>
#include <spdlog/fmt/ostr.h>
#include <af/dim4.hpp>
#include <af/seq.h>
#include <complex>

template<>
struct fmt::formatter<af_seq> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const af_seq& p, FormatContext& ctx) const
        -> decltype(ctx.out()) {
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

#if FMT_VERSION >= 90000
template<>
struct fmt::formatter<af::dim4> : ostream_formatter {};
template<>
struct fmt::formatter<std::complex<float>> : ostream_formatter {};
template<>
struct fmt::formatter<std::complex<double>> : ostream_formatter {};
#endif

template<>
struct fmt::formatter<arrayfire::common::Version> {
    // show major version
    bool show_major = false;
    // show minor version
    bool show_minor = false;
    // show patch version
    bool show_patch = false;

    // Parses format specifications of the form ['M' | 'm' | 'p'].
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        auto it = ctx.begin(), end = ctx.end();
        if (it == end || *it == '}') {
            show_major = show_minor = show_patch = true;
            return it;
        }
        do {
            switch (*it) {
                case 'M': show_major = true; break;
                case 'm': show_minor = true; break;
                case 'p': show_patch = true; break;
                default: throw format_error("invalid format");
            }
            ++it;
        } while (it != end && *it != '}');
        return it;
    }

    // fmt 10 and later require format() to be const, so the parsed flags
    // are combined with the version's contents into locals rather than
    // modified in place.
    template<typename FormatContext>
    auto format(const arrayfire::common::Version& ver, FormatContext& ctx) const
        -> decltype(ctx.out()) {
        if (ver.major() == -1) return format_to(ctx.out(), "N/A");
        const bool minor = show_minor && ver.minor() != -1;
        const bool patch = show_patch && ver.patch() != -1;
        if (show_major && !minor && !patch) {
            return format_to(ctx.out(), "{}", ver.major());
        }
        if (show_major && minor && !patch) {
            return format_to(ctx.out(), "{}.{}", ver.major(), ver.minor());
        }
        if (show_major && minor && patch) {
            return format_to(ctx.out(), "{}.{}.{}", ver.major(), ver.minor(),
                             ver.patch());
        }
        return ctx.out();
    }
};
