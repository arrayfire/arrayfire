/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <cuComplex.h>
#include <string>
#include <af/traits.hpp>

namespace cuda
{
typedef cuFloatComplex   cfloat;
typedef cuDoubleComplex cdouble;
typedef unsigned int   uint;
typedef unsigned char  uchar;
typedef unsigned short ushort;

template<typename T> struct is_complex          { static const bool value = false;  };
template<> struct           is_complex<cfloat>  { static const bool value = true;   };
template<> struct           is_complex<cdouble> { static const bool value = true;   };

template<typename T> const char *shortname(bool caps = true);
template<typename T> const char *getFullName();
}
