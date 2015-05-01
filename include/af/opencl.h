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
#include <af/exception.h>
#include <af/device.h>
#include <stdio.h>

namespace afcl
{
    class array;
   /**
        \addtogroup external
        @{
            @defgroup opencl_mat Interfacing with OpenCL
            How to access ArrayFire's context, queue, and share data with other OpenCL code.

            If your software is using ArrayFire's OpenCL backend, you can also write custom
            kernels and do custom memory operations using native OpenCL commands. The functions
            contained in the `afcl` namespace provide methods to get the context, queue, and
            device(s) that ArrayFire is using as well as convert `cl_mem` handles to
            `af::array` objects.

            Please note: the `af::array` constructors are not thread safe. You may create and
            upload data to `cl_mem` objects from separate threads, but the thread which
            instantiated ArrayFire must do the `cl_mem` to `af::array` conversion.
        @}

    */
    /**
        \ingroup opencl_mat
        @{
    */
    /**
    Get a handle to ArrayFire's OpenCL context

    \param retain If true calls clRetainContext prior to returning the context.
                  Set to true if this value will be passed to a cl::Context constructor
     */
    AFAPI cl_context getContext(bool retain = false);
    /**
    Get a handle to ArrayFire's OpenCL command queue

    \param retain If true calls clRetainCommandQueue prior to returning the context.
                  Set to true if this value will be passed to a cl::CommandQueue constructor
     */
    AFAPI cl_command_queue getQueue(bool retain = false);

    /**
    Get the device ID for ArrayFire's current active device
     */
    AFAPI cl_device_id getDeviceId();

    /**
    Create an af::array object from an OpenCL cl_mem buffer

    \param idims the dimensions of the buffer
    \param type the data type contained in the buffer
    \param retain If true, instructs ArrayFire to retain the memory object. Set to true if
        the memory originates from a cl::Buffer object
     */
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

    /**
    Create an af::array object from an OpenCL cl_mem buffer

    \param dim0 the length of the first dimension of the buffer
    \param type the data type contained in the buffer
    \param retain If true, instructs ArrayFire to retain the memory object. Set to true if
        the memory originates from a cl::Buffer object
     */
    static inline af::array array(dim_type dim0,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0), buf, type, retain);
    }

    /**
    Create an af::array object from an OpenCL cl_mem buffer

    \param dim0 the length of the first dimension of the buffer
    \param dim1 the length of the second dimension of the buffer
    \param type the data type contained in the buffer
    \param retain If true, instructs ArrayFire to retain the memory object. Set to true if
        the memory originates from a cl::Buffer object
     */
    static inline af::array array(dim_type dim0, dim_type dim1,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0, dim1), buf, type, retain);
    }

    /**
    Create an af::array object from an OpenCL cl_mem buffer

    \param dim0 the length of the first dimension of the buffer
    \param dim1 the length of the second dimension of the buffer
    \param dim2 the length of the third dimension of the buffer
    \param type the data type contained in the buffer
    \param retain If true, instructs ArrayFire to retain the memory object. Set to true if
        the memory originates from a cl::Buffer object
     */
    static inline af::array array(dim_type dim0, dim_type dim1,
                                  dim_type dim2,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0, dim1, dim2), buf, type, retain);
    }

    /**
    Create an af::array object from an OpenCL cl_mem buffer

    \param dim0 the length of the first dimension of the buffer
    \param dim1 the length of the second dimension of the buffer
    \param dim2 the length of the third dimension of the buffer
    \param dim3 the length of the fourth dimension of the buffer
    \param type the data type contained in the buffer
    \param retain If true, instructs ArrayFire to retain the memory object. Set to true if
        the memory originates from a cl::Buffer object
     */
    static inline af::array array(dim_type dim0, dim_type dim1,
                                  dim_type dim2, dim_type dim3,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0, dim1, dim2, dim3), buf, type, retain);
    }

    /**
      @}
    */
}
