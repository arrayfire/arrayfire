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

namespace cuda
{
    template<typename T > const char *cuShortName() { return "q"; }
    template<> const char *cuShortName<float   >() { return "f"; }
    template<> const char *cuShortName<double  >() { return "d"; }
    template<> const char *cuShortName<cfloat  >() { return "6float2"; }
    template<> const char *cuShortName<cdouble >() { return "7double2"; }
    template<> const char *cuShortName<int     >() { return "i"; }
    template<> const char *cuShortName<uint    >() { return "j"; }
    template<> const char *cuShortName<char    >() { return "c"; }
    template<> const char *cuShortName<uchar   >() { return "h"; }
    template<> const char *cuShortName<intl    >() { return "x"; }
    template<> const char *cuShortName<uintl   >() { return "y"; }

    template<typename T > const char *afShortName(bool caps) { return caps ?  "Q" : "q"; }
    template<> const char *afShortName<float   >(bool caps) { return caps ?  "S" : "s"; }
    template<> const char *afShortName<double  >(bool caps) { return caps ?  "D" : "d"; }
    template<> const char *afShortName<cfloat  >(bool caps) { return caps ?  "C" : "c"; }
    template<> const char *afShortName<cdouble >(bool caps) { return caps ?  "Z" : "z"; }
    template<> const char *afShortName<int     >(bool caps) { return caps ?  "I" : "i"; }
    template<> const char *afShortName<uint    >(bool caps) { return caps ?  "U" : "u"; }
    template<> const char *afShortName<char    >(bool caps) { return caps ?  "J" : "j"; }
    template<> const char *afShortName<uchar   >(bool caps) { return caps ?  "V" : "v"; }
    template<> const char *afShortName<intl    >(bool caps) { return caps ?  "X" : "x"; }
    template<> const char *afShortName<uintl   >(bool caps) { return caps ?  "Y" : "y"; }

    template<typename T > const char *irname() { return  "i32"; }
    template<> const char *irname<float   >() { return  "float"; }
    template<> const char *irname<double  >() { return  "double"; }
    template<> const char *irname<cfloat  >() { return  "<2 x float>"; }
    template<> const char *irname<cdouble >() { return  "<2 x double>"; }
    template<> const char *irname<int     >() { return  "i32"; }
    template<> const char *irname<uint    >() { return  "i32"; }
    template<> const char *irname<intl    >() { return  "i64"; }
    template<> const char *irname<uintl   >() { return  "i64"; }
    template<> const char *irname<char    >() { return  "i8"; }
    template<> const char *irname<uchar   >() { return  "i8"; }

    template <typename T>
    static inline std::string toString(T val)
    {
        std::stringstream s;
        s << val;
        return s.str();
    }

    template<typename T, bool binary>
    const std::string cuMangledName(const char *fn)
    {
        std::string cname(cuShortName<T>());
        std::string fname(fn);
        size_t flen = fname.size();

        std::string res = std::string("@_Z") + toString(flen) + fname + cname;
        if (binary) {
            if (cname.size() > 1) {
                res = res + "S_";
            } else {
                res = res + cname;
            }
        }
        return res;
    }

#define INSTANTIATE(T)                                                  \
    template const std::string cuMangledName<T, false>(const char *fn); \
    template const std::string cuMangledName<T, true>(const char *fn);  \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(intl)
    INSTANTIATE(uintl)
}
