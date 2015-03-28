/*******************************************************
* Copyright (c) 2014, ArrayFire
* All rights reserved.
*
* This file is distributed under 3-clause BSD license.
* The complete license agreement can be obtained at:
* http://arrayfire.com/licenses/BSD-3-Clause
********************************************************/

#include <af/defines.h>
#include "types.hpp"

namespace opencl
{

    template<typename T > const char *shortname(bool caps) { return caps ? "X" : "x"; }

    template<> const char *shortname<float   >(bool caps) { return caps ? "S" : "s"; }
    template<> const char *shortname<double  >(bool caps) { return caps ? "D" : "d"; }
    template<> const char *shortname<cfloat  >(bool caps) { return caps ? "C" : "c"; }
    template<> const char *shortname<cdouble >(bool caps) { return caps ? "Z" : "z"; }
    template<> const char *shortname<int     >(bool caps) { return caps ? "I" : "i"; }
    template<> const char *shortname<uint    >(bool caps) { return caps ? "U" : "u"; }
    template<> const char *shortname<char    >(bool caps) { return caps ? "J" : "j"; }
    template<> const char *shortname<uchar   >(bool caps) { return caps ? "V" : "v"; }
    template<> const char *shortname<intl    >(bool caps) { return caps ? "L" : "l"; }
    template<> const char *shortname<uintl   >(bool caps) { return caps ? "K" : "k"; }

}
