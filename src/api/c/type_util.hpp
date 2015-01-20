/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <string>
#include <af/defines.h>

const char *getName(af_dtype type);

//uchar to number converters
template<typename T>
struct ToNum
{
    inline T operator()(T val) { return val; }
};

template<>
struct ToNum<unsigned char>
{
    inline int operator()(unsigned char val) { return static_cast<int>(val); }
};

template<>
struct ToNum<char>
{
    inline int operator()(char val) { return static_cast<int>(val); }
};
