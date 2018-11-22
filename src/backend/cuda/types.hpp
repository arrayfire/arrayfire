/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/traits.hpp>
#include <cuComplex.h>

namespace cuda
{
using cdouble = cuDoubleComplex;
using cfloat  = cuFloatComplex;
using uchar   = unsigned char;
using uint    = unsigned int;
using intl    = long long;
using uintl   = unsigned long long;
using ushort  = unsigned short;

template<typename T> struct is_complex          { static const bool value = false;  };
template<> struct           is_complex<cfloat>  { static const bool value = true;   };
template<> struct           is_complex<cdouble> { static const bool value = true;   };

namespace {
template<typename T> const char *shortname(bool caps = false) { return caps ?  "Q" : "q"; }
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
template<> const char *shortname<short   >(bool caps) { return caps ?  "P" : "p"; }
template<> const char *shortname<ushort  >(bool caps) { return caps ?  "Q" : "q"; }

template<typename T> const char *getFullName();

#define SPECIALIZE(T)                                      \
    template<> const char *getFullName<T>() { return #T; }

    SPECIALIZE(float)
    SPECIALIZE(double)
    SPECIALIZE(cfloat)
    SPECIALIZE(cdouble)
    SPECIALIZE(char)
    SPECIALIZE(unsigned char)
    SPECIALIZE(short)
    SPECIALIZE(unsigned short)
    SPECIALIZE(int)
    SPECIALIZE(unsigned int)
    SPECIALIZE(unsigned long long)
    SPECIALIZE(long long)

#undef SPECIALIZE
}

}
