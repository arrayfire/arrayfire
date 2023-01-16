/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/TemplateArg.hpp>
#include <traits.hpp>

#include <string>

template<typename T>
struct TemplateTypename {
    operator TemplateArg() const noexcept {
        return TemplateArg{std::string(af::dtype_traits<T>::getName())};
    }
    operator std::string() const noexcept {
        return std::string(af::dtype_traits<T>::getName());
    }
};

#define SPECIALIZE(TYPE, NAME)                                               \
    template<>                                                               \
    struct TemplateTypename<TYPE> {                                          \
        operator TemplateArg() const noexcept {                              \
            return TemplateArg(std::string(#NAME));                          \
        }                                                                    \
        operator std::string() const noexcept { return std::string(#NAME); } \
    }

SPECIALIZE(unsigned char, detail::uchar);
SPECIALIZE(unsigned int, detail::uint);
SPECIALIZE(unsigned short, detail::ushort);
SPECIALIZE(long long, long long);
SPECIALIZE(unsigned long long, unsigned long long);

#undef SPECIALIZE
