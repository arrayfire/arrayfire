/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <af/defines.h>
#include <af/array.h>
#include <af/exception.h>
#include <af/device.h>
#include <stdio.h>

namespace afcl
{
    AFAPI cl_context getContext(bool retain = false);
    AFAPI cl_command_queue getQueue(bool retain = false);
    AFAPI cl_device_id getDeviceId();

    static inline af::array array(af::dim4 idims, cl_mem buf, af::dtype type, bool retain=false)
    {
        const unsigned ndims = (unsigned)idims.ndims();
        const dim_type *dims = idims.get();

        cl_context context;
        cl_int clerr = clGetMemObjectInfo(buf, CL_MEM_CONTEXT, sizeof(cl_context), &context, NULL);
        if (clerr != CL_SUCCESS) {
            throw af::exception("Failed to get context from cl_mem object \"buf\" ");
        }

        if (context != getContext()) {
            throw(af::exception("Context mismatch between input \"buf\" and arrayfire"));
        }


        if (retain) clerr = clRetainMemObject(buf);

        af_array out;
        af_err err = af_device_array(&out, buf, ndims, dims, type);

        if (err != AF_SUCCESS || clerr != CL_SUCCESS) {
            if (retain && clerr == CL_SUCCESS) clReleaseMemObject(buf);
            throw af::exception("Failed to create device array");
        }

        return af::array(out);
    }

    static inline af::array array(dim_type dim0,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0), buf, type, retain);
    }

    static inline af::array array(dim_type dim0, dim_type dim1,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0, dim1), buf, type, retain);
    }

    static inline af::array array(dim_type dim0, dim_type dim1,
                                  dim_type dim2,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0, dim1, dim2), buf, type, retain);
    }

    static inline af::array array(dim_type dim0, dim_type dim1,
                                  dim_type dim2, dim_type dim3,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0, dim1, dim2, dim3), buf, type, retain);
    }
}
