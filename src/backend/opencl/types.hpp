#pragma once
#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace opencl
{
    typedef cl_float2   cfloat;
    typedef cl_double2 cdouble;
    typedef cl_uchar     uchar;
    typedef cl_uint       uint;

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
}
