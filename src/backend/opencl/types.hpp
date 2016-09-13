/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#pragma GCC diagnostic pop

namespace opencl
{

    typedef cl_float2   cfloat;
    typedef cl_double2 cdouble;
    typedef cl_uchar     uchar;
    typedef cl_uint       uint;
    typedef cl_ushort   ushort;

    template<typename T> struct is_complex          { static const bool value = false;  };
    template<> struct           is_complex<cfloat>  { static const bool value = true;   };
    template<> struct           is_complex<cdouble> { static const bool value = true;   };

    template<typename T > const char *shortname(bool caps=false);

}
