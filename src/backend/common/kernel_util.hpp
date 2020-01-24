/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <traits.hpp>

#include <string>

template<typename T>
std::string toString(T value);

struct TemplateArg {
    std::string _tparam;

    TemplateArg(std::string str) : _tparam(str) {}

    template<typename T>
    constexpr TemplateArg(T value) noexcept : _tparam(toString(value)) {}
};

template<typename T>
struct TemplateTypename {
    operator TemplateArg() const noexcept {
        return {std::string(dtype_traits<T>::getName())};
    }
};

#define SPECIALIZE(TYPE, NAME)                      \
    template<>                                      \
    struct TemplateTypename<TYPE> {                 \
        operator TemplateArg() const noexcept {     \
            return TemplateArg(std::string(#NAME)); \
        }                                           \
    }

SPECIALIZE(unsigned char, detail::uchar);
SPECIALIZE(unsigned int, detail::uint);
SPECIALIZE(unsigned short, detail::ushort);
SPECIALIZE(long long, long long);
SPECIALIZE(unsigned long long, unsigned long long);

#undef SPECIALIZE

#define DefineKey(arg) " -D " #arg
#define DefineValue(arg) " -D " #arg "=" + toString(arg)
#define DefineKeyValue(key, arg) " -D " #key "=" + toString(arg)
