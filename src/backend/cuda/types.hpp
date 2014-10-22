#pragma once
#include <cuComplex.h>

namespace cuda
{
    typedef cuFloatComplex   cfloat;
    typedef cuDoubleComplex cdouble;
    typedef unsigned int   uint;
    typedef unsigned char uchar;

    template<typename T> struct is_complex          { static const bool value = false;  };
    template<> struct           is_complex<cfloat>  { static const bool value = true;   };
    template<> struct           is_complex<cdouble> { static const bool value = true;   };

    template<typename T > static const char *shortname(bool caps) { return caps ? "X" : "x"; }
    template<> const char *shortname<float   >(bool caps) { return caps ? "S" : "s"; }
    template<> const char *shortname<double  >(bool caps) { return caps ? "D" : "d"; }
    template<> const char *shortname<cfloat  >(bool caps) { return caps ? "C" : "c"; }
    template<> const char *shortname<cdouble >(bool caps) { return caps ? "Z" : "z"; }
    template<> const char *shortname<int     >(bool caps) { return caps ? "I" : "i"; }
    template<> const char *shortname<uint    >(bool caps) { return caps ? "U" : "u"; }
    template<> const char *shortname<char    >(bool caps) { return caps ? "J" : "j"; }
    template<> const char *shortname<uchar   >(bool caps) { return caps ? "V" : "v"; }

    template<typename T > static const char *irname() { return  "i32"; }
    template<> const char *irname<float   >() { return  "float"; }
    template<> const char *irname<double  >() { return  "double"; }
    template<> const char *irname<cfloat  >() { return  "<2 x float>"; }
    template<> const char *irname<cdouble >() { return  "<2 x double>"; }
    template<> const char *irname<int     >() { return  "i32"; }
    template<> const char *irname<uint    >() { return  "i32"; }
    template<> const char *irname<char    >() { return  "i8"; }
    template<> const char *irname<uchar   >() { return  "i8"; }
}
