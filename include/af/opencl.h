/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <af/defines.h>

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 33
typedef enum
{
    AFCL_DEVICE_TYPE_CPU     = CL_DEVICE_TYPE_CPU,
    AFCL_DEVICE_TYPE_GPU     = CL_DEVICE_TYPE_GPU,
    AFCL_DEVICE_TYPE_ACC     = CL_DEVICE_TYPE_ACCELERATOR,
    AFCL_DEVICE_TYPE_UNKNOWN = -1
} afcl_device_type;
#endif

#if AF_API_VERSION >= 33
typedef enum
{
    AFCL_PLATFORM_AMD     = 0,
    AFCL_PLATFORM_APPLE   = 1,
    AFCL_PLATFORM_INTEL   = 2,
    AFCL_PLATFORM_NVIDIA  = 3,
    AFCL_PLATFORM_BEIGNET = 4,
    AFCL_PLATFORM_POCL    = 5,
    AFCL_PLATFORM_UNKNOWN = -1
} afcl_platform;
#endif

/**
    \ingroup opencl_mat
    @{
*/
/**
  Get a handle to ArrayFire's OpenCL context

  \param[out] ctx the current context being used by ArrayFire
  \param[in] retain if true calls clRetainContext prior to returning the context
  \returns \ref af_err error code

  \note Set \p retain to true if this value will be passed to a cl::Context constructor
*/
AFAPI af_err afcl_get_context(cl_context *ctx, const bool retain);

/**
  Get a handle to ArrayFire's OpenCL command queue

  \param[out] queue the current command queue being used by ArrayFire
  \param[in] retain if true calls clRetainCommandQueue prior to returning the context
  \returns \ref af_err error code

  \note Set \p retain to true if this value will be passed to a cl::CommandQueue constructor
*/
AFAPI af_err afcl_get_queue(cl_command_queue *queue, const bool retain);

/**
   Get the device ID for ArrayFire's current active device

   \param[out] id the cl_device_id of the current device
   \returns \ref af_err error code
*/
AFAPI af_err afcl_get_device_id(cl_device_id *id);

#if AF_API_VERSION >= 32
/**
   Set ArrayFire's active device based on \p id of type cl_device_id

   \param[in] id the cl_device_id of the device to be set as active device
   \returns \ref af_err error code
*/
AFAPI af_err afcl_set_device_id(cl_device_id id);
#endif

#if AF_API_VERSION >= 33
/**
   Push user provided device control constructs into the ArrayFire device manager pool

   This function should be used only when the user would like ArrayFire to use an
   user generated OpenCL context and related objects for ArrayFire operations.

   \param[in] dev is the OpenCL device for which user provided context will be used by ArrayFire
   \param[in] ctx is the user provided OpenCL cl_context to be used by ArrayFire
   \param[in] que is the user provided OpenCL cl_command_queue to be used by ArrayFire. If this
                  parameter is NULL, then we create a command queue for the user using the OpenCL
                  context they provided us.

   \note ArrayFire does not take control of releasing the objects passed to it. The user needs to release them appropriately.
*/
AFAPI af_err afcl_add_device_context(cl_device_id dev, cl_context ctx, cl_command_queue que);
#endif

#if AF_API_VERSION >= 33
/**
   Set active device using cl_context and cl_device_id

   \param[in] dev is the OpenCL device id that is to be set as Active device inside ArrayFire
   \param[in] ctx is the OpenCL cl_context being used by ArrayFire
*/
AFAPI af_err afcl_set_device_context(cl_device_id dev, cl_context ctx);
#endif

#if AF_API_VERSION >= 33
/**
   Remove the user provided device control constructs from the ArrayFire device manager pool

   This function should be used only when the user would like ArrayFire to remove an already
   pushed user generated OpenCL context and related objects.

   \param[in] dev is the OpenCL device id that has to be popped
   \param[in] ctx is the cl_context object to be removed from ArrayFire pool

   \note ArrayFire does not take control of releasing the objects passed to it. The user needs to release them appropriately.
*/
AFAPI af_err afcl_delete_device_context(cl_device_id dev, cl_context ctx);
#endif

#if AF_API_VERSION >= 33
/**
   Get the type of the current device
*/
AFAPI af_err afcl_get_device_type(afcl_device_type *res);
#endif

#if AF_API_VERSION >= 33
/**
   Get the platform of the current device
*/
AFAPI af_err afcl_get_platform(afcl_platform *res);
#endif

/**
  @}
*/

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
     \addtogroup opencl_mat
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

#if AF_API_VERSION >= 32
 /**
   Set ArrayFire's active device based on \p id of type cl_device_id

   \param[in] id the cl_device_id of the device to be set as active device
 */
 static inline void setDeviceId(cl_device_id id)
 {
     af_err err = afcl_set_device_id(id);
     if (err != AF_SUCCESS) throw af::exception("Failed to set OpenCL device as active device");
 }
#endif

#if AF_API_VERSION >= 33
/**
   Push user provided device control constructs into the ArrayFire device manager pool

   This function should be used only when the user would like ArrayFire to use an
   user generated OpenCL context and related objects for ArrayFire operations.

   \param[in] dev is the OpenCL device for which user provided context will be used by ArrayFire
   \param[in] ctx is the user provided OpenCL cl_context to be used by ArrayFire
   \param[in] que is the user provided OpenCL cl_command_queue to be used by ArrayFire. If this
                  parameter is NULL, then we create a command queue for the user using the OpenCL
                  context they provided us.

   \note ArrayFire does not take control of releasing the objects passed to it. The user needs to release them appropriately.
*/
static inline void addDevice(cl_device_id dev, cl_context ctx, cl_command_queue que)
{
    af_err err = afcl_add_device_context(dev, ctx, que);
    if (err!=AF_SUCCESS) throw af::exception("Failed to push user provided device/context to ArrayFire pool");
}
#endif

#if AF_API_VERSION >= 33
/**
   Set active device using cl_context and cl_device_id

   \param[in] dev is the OpenCL device id that is to be set as Active device inside ArrayFire
   \param[in] ctx is the OpenCL cl_context being used by ArrayFire
*/
static inline void setDevice(cl_device_id dev, cl_context ctx)
{
    af_err err = afcl_set_device_context(dev, ctx);
    if (err!=AF_SUCCESS) throw af::exception("Failed to set device based on cl_device_id & cl_context");
}
#endif

#if AF_API_VERSION >= 33
/**
   Remove the user provided device control constructs from the ArrayFire device manager pool

   This function should be used only when the user would like ArrayFire to remove an already
   pushed user generated OpenCL context and related objects.

   \param[in] dev is the OpenCL device id that has to be popped
   \param[in] ctx is the cl_context object to be removed from ArrayFire pool

   \note ArrayFire does not take control of releasing the objects passed to it. The user needs to release them appropriately.
*/
static inline void deleteDevice(cl_device_id dev, cl_context ctx)
{
    af_err err = afcl_delete_device_context(dev, ctx);
    if (err!=AF_SUCCESS) throw af::exception("Failed to remove the requested device from ArrayFire device pool");
}
#endif


#if AF_API_VERSION >= 33
 typedef afcl_device_type deviceType;
 typedef afcl_platform platform;
#endif

#if AF_API_VERSION >= 33
/**
   Get the type of the current device
*/
static inline deviceType getDeviceType()
{
    afcl_device_type res = AFCL_DEVICE_TYPE_UNKNOWN;
    af_err err = afcl_get_device_type(&res);
    if (err!=AF_SUCCESS) throw af::exception("Failed to get OpenCL device type");
    return res;
}
#endif

#if AF_API_VERSION >= 33
/**
   Get a vendor enumeration for the current platform
*/
static inline platform getPlatform()
{
    afcl_platform res = AFCL_PLATFORM_UNKNOWN;
    af_err err = afcl_get_platform(&res);
    if (err!=AF_SUCCESS) throw af::exception("Failed to get OpenCL platform");
    return res;
}
#endif

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


#endif
