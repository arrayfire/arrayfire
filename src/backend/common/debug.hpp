/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <boost/stacktrace.hpp>
#include <common/ArrayFireTypesIO.hpp>
#include <common/jit/NodeIO.hpp>
#include <spdlog/fmt/bundled/format.h>
#include <iostream>

#define DBGTRACE(msg)                                              \
    fmt::print(std::cout, __FILE__ ":{}:{}\n{}\n", __LINE__, #msg, \
               boost::stacktrace::stacktrace())

namespace debugging {

template<typename first>
void print(const char *F, const first &FF) {
    fmt::print(std::cout, "{} = {}", F, FF);
}

template<typename first, typename... ARGS>
void print(const char *F, const first &FF, ARGS... args) {
    fmt::print(std::cout, "{} = {} | ", F, FF);
    print(args...);
}
}  // namespace debugging

#define SHOW1(val1) debugging::print(#val1, val1)
#define SHOW2(val1, val2) debugging::print(#val1, val1, #val2, val2)
#define SHOW3(val1, val2, val3) \
    debugging::print(#val1, val1, #val2, val2, #val3, val3)

#define SHOW4(val1, val2, val3, val4) \
    debugging::print(#val1, val1, #val2, val2, #val3, val3, #val4, val4)
#define SHOW5(val1, val2, val3, val4, val5)                              \
    debugging::print(#val1, val1, #val2, val2, #val3, val3, #val4, val4, \
                     #val5, val5)
#define SHOW6(val1, val2, val3, val4, val5, val6)                        \
    debugging::print(#val1, val1, #val2, val2, #val3, val3, #val4, val4, \
                     #val5, val5, #val6, val6)

#define GET_MACRO(_1, _2, _3, _4, _5, _6, NAME, ...) NAME

#define SHOW(...)                                                        \
    do {                                                                 \
        fmt::print(std::cout, "{}:({}): ", __FILE__, __LINE__);          \
        GET_MACRO(__VA_ARGS__, SHOW6, SHOW5, SHOW4, SHOW3, SHOW2, SHOW1) \
        (__VA_ARGS__);                                                   \
        fmt::print(std::cout, "\n");                                     \
    } while (0)

#define PRINTVEC(val)                                                        \
    do {                                                                     \
        fmt::print(std::cout, "{}:({}):{} [{}]\n", __FILE__, __LINE__, #val, \
                   fmt::join(val, ", "));                                    \
    } while (0)
