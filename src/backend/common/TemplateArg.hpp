/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <string>
#include <utility>

template<typename T>
std::string toString(T value);

struct TemplateArg {
    std::string _tparam;

    TemplateArg(std::string str) : _tparam(std::move(str)) {}

    template<typename T>
    constexpr TemplateArg(T value) noexcept : _tparam(toString(value)) {}
};

#define DefineKey(arg) " -D " #arg
#define DefineValue(arg) " -D " #arg "=" + toString(arg)
#define DefineKeyValue(key, arg) " -D " #key "=" + toString(arg)
