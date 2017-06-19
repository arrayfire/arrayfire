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
#include <sstream>
#include <af/traits.hpp>

namespace cuda
{

    template<typename T > const char *shortname(bool caps) { return caps ?  "Q" : "q"; }
    template<> const char *shortname<float   >(bool caps) { return caps ?  "S" : "s"; }
    template<> const char *shortname<double  >(bool caps) { return caps ?  "D" : "d"; }
    template<> const char *shortname<cfloat  >(bool caps) { return caps ?  "C" : "c"; }
    template<> const char *shortname<cdouble >(bool caps) { return caps ?  "Z" : "z"; }
    template<> const char *shortname<int     >(bool caps) { return caps ?  "I" : "i"; }
    template<> const char *shortname<uint    >(bool caps) { return caps ?  "U" : "u"; }
    template<> const char *shortname<char    >(bool caps) { return caps ?  "J" : "j"; }
    template<> const char *shortname<uchar   >(bool caps) { return caps ?  "V" : "v"; }
    template<> const char *shortname<intl    >(bool caps) { return caps ?  "X" : "x"; }
    template<> const char *shortname<uintl   >(bool caps) { return caps ?  "Y" : "y"; }
    template<> const char *shortname<short   >(bool caps) { return caps ?  "P" : "P"; }
    template<> const char *shortname<ushort  >(bool caps) { return caps ?  "Q" : "Q"; }

#define INSTANTIATE(T)                                      \
    template<> const char *getFullName<T>() { return #T; }  \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(char)
    INSTANTIATE(unsigned char)
    INSTANTIATE(short)
    INSTANTIATE(unsigned short)
    INSTANTIATE(int)
    INSTANTIATE(unsigned int)
    INSTANTIATE(unsigned long long)
    INSTANTIATE(long long)
}
