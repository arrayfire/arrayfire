/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifdef AF_ONEAPI
#include <sycl/sycl.hpp>
#else

#endif

#include <af/defines.h>
#include <af/traits.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 39
typedef enum {
  AF_SYCL_DEVICE_TYPE_HOST = (1 << 0),
  AF_SYCL_DEVICE_TYPE_CPU = (1 << 1),
  AF_SYCL_DEVICE_TYPE_GPU = (1 << 2),
  AF_SYCL_DEVICE_TYPE_ACC = (1 << 3),
  AF_SYCL_DEVICE_TYPE_CUSTOM = (1 << 4),
  AF_SYCL_DEVICE_TYPE_AUTOMATIC = (1 << 5),
  AF_SYCL_DEVICE_TYPE_ALL = 0xFFFFFFF,
  AF_SYCL_DEVICE_TYPE_UNKNOWN = -1
} afsycl_device_type;

typedef enum {
  AF_SYCL_PLATFORM_AMD = 0,
  AF_SYCL_PLATFORM_APPLE = 1,
  AF_SYCL_PLATFORM_INTEL = 2,
  AF_SYCL_PLATFORM_NVIDIA = 3,
  AF_SYCL_PLATFORM_BEIGNET = 4,
  AF_SYCL_PLATFORM_POCL = 5,
  AF_SYCL_PLATFORM_UNKNOWN = -1
} afsycl_platform;

/**
    \ingroup oneapi_mat
    @{
*/
/**
  Get a handle to ArrayFire's sycl context

  \param[out] ctx the current context being used by ArrayFire
  \returns \ref af_err error code

  \note Set \p retain to true if this value will be passed to a sycl::context
  constructor
*/
AFAPI af_err afsycl_get_context(sycl::context *ctx);

/**
  Get a handle to ArrayFire's sycl queue

  \param[out] queue the current command queue being used by ArrayFire
  \returns \ref af_err error code
*/
AFAPI af_err afsycl_get_queue(sycl::queue *queue);

/**
   Get the native device for ArrayFire's current active device

   \param[out] dev the sycl::device corresponding to the current device
   \returns \ref af_err error code
*/
AFAPI af_err afsycl_get_device(sycl::device *dev);

/**
   Set ArrayFire's active device to \p dev

   \param[in] dev the device to be set as active device
   \returns \ref af_err error code
*/
AFAPI af_err afsycl_set_device(sycl::device dev);

/**
   Push user provided device control constructs into the ArrayFire device
   manager pool This function will infer the sycl::device and sycl::context
   correspondning to the provided sycl::queue

   This function should be used only when the user would like ArrayFire to use
   an user generated sycl context and related objects for ArrayFire operations.

   \param[in] que is the user provided sycl queue to be used by ArrayFire
*/
AFAPI af_err afsycl_add_queue(sycl::queue que);

/**
   Set active device using sycl::context and sycl::device

   \param[in] dev is the sycl device that is to be set as Active device inside
   ArrayFire \param[in] ctx is the sycl context being used by ArrayFire
*/
AFAPI af_err afsycl_set_device_context(sycl::device dev, sycl::context ctx);

/**
   Remove the user provided device control constructs from the ArrayFire device
   manager pool

   This function should be used only when the user would like ArrayFire to
   remove an already pushed user generated sycl context and related objects.

   \param[in] dev is the sycl device that has to be popped
   \param[in] ctx is the sycl context object to be removed from ArrayFire pool

   \note ArrayFire does not take control of releasing the objects passed to it.
   The user needs to release them appropriately.
*/
AFAPI af_err afsycl_delete_device_context(sycl::device dev, sycl::context ctx);

/*
 Get the type of the current device
*/
AFAPI af_err afsycl_get_device_type(afsycl_device_type *res);

/**
   Get the platform of the current device
*/
AFAPI af_err afsycl_get_platform(afsycl_platform *res);
#endif

/**
  @}
*/

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <af/array.h>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/exception.h>
#include <stdio.h>

namespace afsycl {

#if AF_API_VERSION >= 39
/**
    \addtogroup oneapi_mat
    @{
*/

/**
Get a handle to ArrayFire's sycl context

\param[in] retain if true calls clRetainContext prior to returning the context
\returns the current context being used by ArrayFire

*/
static inline sycl::context getContext() {
  sycl::context ctx;
  af_err err = afsycl_get_context(&ctx);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to get sycl context from arrayfire");
  return ctx;
}

/**
Get a handle to ArrayFire's sycl command queue

\returns the current command queue being used by ArrayFire

*/
static inline sycl::queue getQueue() {
  sycl::queue q;
  af_err err = afsycl_get_queue(&q);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to get sycl queue from arrayfire");
  return q;
}

/**
   Get the device ID for ArrayFire's current active device
   \returns the cl_device_id of the current device
*/
static inline sycl::device getDeviceId() {
  sycl::device dev;
  af_err err = afsycl_get_device(&dev);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to get sycl device");

  return dev;
}

/**
  Set ArrayFire's active device to sycl::device

  \param[in] dev the sycl device to be set as active device
*/
static inline void setDeviceId(sycl::device dev) {
  af_err err = afsycl_set_device(dev);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to set sycl device as active device");
}

/**
   Push user provided device control constructs into the ArrayFire device
   manager pool This function will infer the sycl::device and sycl::context
   correspondning to the provided sycl::queue

   This function should be used only when the user would like ArrayFire to use
   an user generated sycl context and related objects for ArrayFire operations.

   \param[in] que is the user provided sycl queue to be used by ArrayFire
*/
static inline void addQueue(sycl::queue que) {
  af_err err = afsycl_add_queue(que);
  if (err != AF_SUCCESS)
    throw af::exception(
        "Failed to push user provided queue/device/context to ArrayFire pool");
}

/**
   Set active device using sycl::context and sycl::device

   \param[in] dev is the sycl device id that is to be set as Active device
   inside ArrayFire \param[in] ctx is the sycl context being used by ArrayFire
*/
static inline void setDevice(sycl::device dev, sycl::context ctx) {
  af_err err = afsycl_set_device_context(dev, ctx);
  if (err != AF_SUCCESS)
    throw af::exception(
        "Failed to set device based on sycl::device & sycl::context");
}

/**
   Remove the user provided device control constructs from the ArrayFire device
   manager pool

   This function should be used only when the user would like ArrayFire to
   remove an already pushed user generated sycl context and related objects.

   \param[in] dev is the sycl device id that has to be popped
   \param[in] ctx is the context object to be removed from ArrayFire pool

   \note ArrayFire does not take control of releasing the objects passed to it.
   The user needs to release them appropriately.
*/
static inline void deleteDevice(sycl::device dev, sycl::context ctx) {
  af_err err = afsycl_delete_device_context(dev, ctx);
  if (err != AF_SUCCESS)
    throw af::exception(
        "Failed to remove the requested device from ArrayFire device pool");
}

typedef afsycl_device_type deviceType;
typedef afsycl_platform platform;

/**
   Get the type of the current device
*/
static inline deviceType getDeviceType() {
  afsycl_device_type res = AF_SYCL_DEVICE_TYPE_UNKNOWN;
  af_err err = afsycl_get_device_type(&res);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to get sycl device type");
  return res;
}

/**
   Get a vendor enumeration for the current platform
*/
static inline platform getPlatform() {
  afsycl_platform res = AF_SYCL_PLATFORM_UNKNOWN;
  af_err err = afsycl_get_platform(&res);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to get sycl platform");
  return res;
}

/**
Create an af::array object from a sycl::buffer

\param[in] idims the dimensions of the buffer
\param[in] buf the sycl memory object
\param[in] type the data type contained in the buffer
\returns an array object created from the sycl buffer

\note Set \p retain to true if the memory originates from a cl::Buffer object
 */
template <typename T>
static inline af::array array(af::dim4 idims, sycl::buffer<T> buf) {
  af::dtype type = (af::dtype)af::dtype_traits<T>::af_type;
  const unsigned ndims = (unsigned)idims.ndims();
  const dim_t *dims = idims.get();

  /*
      if(!buf.template has_property<sycl::property::buffer::context_bound>()) {
              throw af::exception("Failed to get context from sycl::buffer
  object \"buf\" ");
      }
  sycl::context ctx = buf.template
  get_property<sycl::property::buffer::context_bound>().get_context();

      if (ctx != getContext()) {
              throw(af::exception("Context mismatch between input \"buf\" and
  arrayfire"));
      }
  */

  af_array out;
  af_err err = af_device_array(&out, &buf, ndims, dims, type);

  if (err != AF_SUCCESS) {
    throw af::exception("Failed to create device array");
  }
  return af::array(out);
}

/**
Create an af::array object from a sycl buffer

\param[in] dim0 the length of the first dimension of the buffer
\param[in] buf the sycl::buffer
\returns an array object created from the sycl buffer
 */
template <typename T>
static inline af::array array(dim_t dim0, sycl::buffer<T> buf) {
  return afsycl::array(af::dim4(dim0), buf);
}

/**
Create an af::array object from a sycl::buffer

\param[in] dim0 the length of the first dimension of the buffer
\param[in] dim1 the length of the second dimension of the buffer
\param[in] buf the sycl::buffer
\returns an array object created from the sycl buffer

 */
template <typename T>
static inline af::array array(dim_t dim0, dim_t dim1, sycl::buffer<T> buf) {
  return afsycl::array(af::dim4(dim0, dim1), buf);
}

/**
Create an af::array object from a sycl::buffer

\param[in] dim0 the length of the first dimension of the buffer
\param[in] dim1 the length of the second dimension of the buffer
\param[in] dim2 the length of the third dimension of the buffer
\param[in] buf the sycl::buffer
\returns an array object created from the sycl buffer

 */
template <typename T>
static inline af::array array(dim_t dim0, dim_t dim1, dim_t dim2,
                              sycl::buffer<T> buf) {
  return afsycl::array(af::dim4(dim0, dim1, dim2), buf, type, retain);
}

/**
Create an af::array object from a sycl::buffer

\param[in] dim0 the length of the first dimension of the buffer
\param[in] dim1 the length of the second dimension of the buffer
\param[in] dim2 the length of the third dimension of the buffer
\param[in] dim3 the length of the fourth dimension of the buffer
\param[in] buf the sycl::buffer
\returns an array object created from the sycl buffer

 */
template <typename T>
static inline af::array array(dim_t dim0, dim_t dim1, dim_t dim2, dim_t dim3,
                              sycl::buffer<T> buf) {
  return afsycl::array(af::dim4(dim0, dim1, dim2, dim3), buf);
}

/**
   @}
*/
#endif

} // namespace afsycl

#endif
