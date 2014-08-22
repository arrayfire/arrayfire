#pragma once
#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace opencl {
    typedef cl_float2   cfloat;
    typedef cl_double2 cdouble;
    typedef cl_uchar     uchar;
    typedef cl_uint       uint;
}

namespace detail = opencl;
