/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/util.hpp>

#include <array>
#include <string>
#include <utility>

template<typename T>
struct TemplateTypename;

struct TemplateArg {
    std::string _tparam;

    TemplateArg(std::string str) : _tparam(std::move(str)) {}

    template<typename T>
    constexpr TemplateArg(TemplateTypename<T> arg) noexcept : _tparam(arg) {}

    template<typename T>
    constexpr TemplateArg(T value) noexcept
        : _tparam(arrayfire::common::toString(value)) {}
};

template<typename... Targs>
std::array<TemplateArg, sizeof...(Targs)> TemplateArgs(Targs &&...args) {
    return std::array<TemplateArg, sizeof...(Targs)>{
        std::forward<Targs>(args)...};
}

#define DefineKey(arg) " -D " #arg
#define DefineValue(arg) " -D " #arg "=" + arrayfire::common::toString(arg)
#define DefineKeyValue(key, arg) \
    " -D " #key "=" + arrayfire::common::toString(arg)
#define DefineKeyFromStr(arg) " -D " + std::string(arg)
