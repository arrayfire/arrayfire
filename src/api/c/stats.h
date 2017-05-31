/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

template<typename T, typename Other>
struct is_same{
    static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
    static const bool value = true;
};

template<bool, typename T, typename O>
struct cond_type;

template<typename T, typename Other>
struct cond_type<true, T, Other> {
    typedef T type;
};

template<typename T, typename Other>
struct cond_type<false, T, Other> {
    typedef Other type;
};

template<typename T>
struct baseOutType {
    typedef typename cond_type< is_same<T, cdouble>::value ||
                                is_same<T, double>::value,
                                double,
                                float>::type type;
};
