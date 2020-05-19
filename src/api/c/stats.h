/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <backend.hpp>
#include <type_traits>

template<typename T>
struct baseOutType {
    typedef typename std::conditional<std::is_same<T, detail::cdouble>::value ||
                                          std::is_same<T, double>::value,
                                      double, float>::type type;
};
