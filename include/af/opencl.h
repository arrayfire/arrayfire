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

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err afcl_get_context(cl_context *ctx, const bool retain);

    AFAPI af_err afcl_get_queue(cl_command_queue *queue, const bool retain);

    AFAPI af_err afcl_get_device_id(cl_device_id *id);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <af/array.h>
#include <af/dim4.hpp>
#include <af/exception.h>
#include <af/device.h>
#include <stdio.h>

namespace afcl
{
   /**

    */
    /**
        \ingroup opencl_mat
        @{
    */
    /**
    Get a handle to ArrayFire's OpenCL context

    \param[in] retain if true calls clRetainContext prior to returning the context
    \returns the current context being used by ArrayFire

    \note Set \p retain to true if this value will be passed to a cl::Context constructor
    */
    static inline cl_context getContext(bool retain = false)
    {
        cl_context ctx;
        af_err err = afcl_get_context(&ctx, retain);
        if (err != AF_SUCCESS) throw af::exception("Failed to get OpenCL context from arrayfire");
        return ctx;
    }

    /**
    Get a handle to ArrayFire's OpenCL command queue

    \param[in] retain if true calls clRetainCommandQueue prior to returning the context
    \returns the current command queue being used by ArrayFire

    \note Set \p retain to true if this value will be passed to a cl::CommandQueue constructor
    */
    static inline cl_command_queue getQueue(bool retain = false)
    {
        cl_command_queue queue;
        af_err err = afcl_get_queue(&queue, retain);
        if (err != AF_SUCCESS) throw af::exception("Failed to get OpenCL command queue from arrayfire");
        return queue;
    }

    /**
       Get the device ID for ArrayFire's current active device
       \returns the cl_device_id of the current device
    */
    static inline cl_device_id getDeviceId()
    {
        cl_device_id id;
        af_err err = afcl_get_device_id(&id);
        if (err != AF_SUCCESS) throw af::exception("Failed to get OpenCL device ID");

        return id;
    }

    /**
    Create an af::array object from an OpenCL cl_mem buffer

    \param[in] idims the dimensions of the buffer
    \param[in] buf the OpenCL memory object
    \param[in] type the data type contained in the buffer
    \param[in] retain if true, instructs ArrayFire to retain the memory object
    \returns an array object created from the OpenCL buffer

    \note Set \p retain to true if the memory originates from a cl::Buffer object
     */
    static inline af::array array(af::dim4 idims, cl_mem buf, af::dtype type, bool retain=false)
    {
        const unsigned ndims = (unsigned)idims.ndims();
        const dim_t *dims = idims.get();

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

    \param[in] dim0 the length of the first dimension of the buffer
    \param[in] buf the OpenCL memory object
    \param[in] type the data type contained in the buffer
    \param[in] retain if true, instructs ArrayFire to retain the memory object
    \returns an array object created from the OpenCL buffer

    \note Set \p retain to true if the memory originates from a cl::Buffer object
     */
    static inline af::array array(dim_t dim0,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0), buf, type, retain);
    }

    /**
    Create an af::array object from an OpenCL cl_mem buffer

    \param[in] dim0 the length of the first dimension of the buffer
    \param[in] dim1 the length of the second dimension of the buffer
    \param[in] buf the OpenCL memory object
    \param[in] type the data type contained in the buffer
    \param[in] retain if true, instructs ArrayFire to retain the memory object
    \returns an array object created from the OpenCL buffer

    \note Set \p retain to true if the memory originates from a cl::Buffer object
     */
    static inline af::array array(dim_t dim0, dim_t dim1,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0, dim1), buf, type, retain);
    }

    /**
    Create an af::array object from an OpenCL cl_mem buffer

    \param[in] dim0 the length of the first dimension of the buffer
    \param[in] dim1 the length of the second dimension of the buffer
    \param[in] dim2 the length of the third dimension of the buffer
    \param[in] buf the OpenCL memory object
    \param[in] type the data type contained in the buffer
    \param[in] retain if true, instructs ArrayFire to retain the memory object
    \returns an array object created from the OpenCL buffer

    \note Set \p retain to true if the memory originates from a cl::Buffer object
     */
    static inline af::array array(dim_t dim0, dim_t dim1,
                                  dim_t dim2,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0, dim1, dim2), buf, type, retain);
    }

    /**
    Create an af::array object from an OpenCL cl_mem buffer

    \param[in] dim0 the length of the first dimension of the buffer
    \param[in] dim1 the length of the second dimension of the buffer
    \param[in] dim2 the length of the third dimension of the buffer
    \param[in] dim3 the length of the fourth dimension of the buffer
    \param[in] buf the OpenCL memory object
    \param[in] type the data type contained in the buffer
    \param[in] retain if true, instructs ArrayFire to retain the memory object
    \returns an array object created from the OpenCL buffer

    \note Set \p retain to true if the memory originates from a cl::Buffer object
     */
    static inline af::array array(dim_t dim0, dim_t dim1,
                                  dim_t dim2, dim_t dim3,
                                  cl_mem buf, af::dtype type, bool retain=false)
    {
        return afcl::array(af::dim4(dim0, dim1, dim2, dim3), buf, type, retain);
    }

    /**
      @}
    */
}

namespace af {
    template<> AFAPI cl_mem *array::device() const
    {
        cl_mem *mem = new cl_mem;
        af_err err = af_get_device_ptr((void **)mem, get());
        if (err != AF_SUCCESS) throw af::exception("Failed to get cl_mem from array object");
        return mem;
    }
}

#endif
