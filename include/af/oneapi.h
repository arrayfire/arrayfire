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
// users will need to cast return types to expected
// sycl:: datatypes when using sycl with the C-api
typedef void *af_sycl_context;
typedef void *af_sycl_queue;
typedef void *af_sycl_device;
typedef void *af_sycl_device_type;
typedef void *af_sycl_platform;

/**
    \ingroup oneapi_matj
    @{
*/
/**
  Get a handle to ArrayFire's sycl context

  \param[out] ctx the current context being used by ArrayFire
              will need to be cast to sycl::context*
  \returns \ref af_err error code

  \note Set \p retain to true if this value will be passed to a sycl::context
  constructor
*/
AFAPI af_err afoneapi_get_context(af_sycl_context ctx);

/**
  Get a handle to ArrayFire's sycl queue

  \param[out] queue the current command queue being used by ArrayFire,
              will need to be cast to sycl::queue*
  \returns \ref af_err error code
*/
AFAPI af_err afoneapi_get_queue(af_sycl_queue queue);

/**
   Get a handle to the native device for ArrayFire's current active device

   \param[out] dev the sycl::device corresponding to the current device
               will need to be cast to sycl::device*
   \returns \ref af_err error code
*/
AFAPI af_err afoneapi_get_device(af_sycl_device dev);

/*
    Get a handle to the device_type of the current active device
    \param[out] devtype the sycl::info::device_type corresponding to the current
   device will need to be cast to sycl::info::device_type* \returns \ref af_err
   error code
*/
AFAPI af_err afoneapi_get_device_type(af_sycl_device_type devtype);

/**
   Get a handle to the platform of the current active device
    \param[out] plat the platform corresponding to the current device
                will need to be cast to sycl::platform*
        \returns \ref af_err error code
*/
AFAPI af_err afoneapi_get_platform(af_sycl_platform plat);

/**
   Set ArrayFire's active device to \p dev. If device does not currently
   exist within the current available devices it will be added to the
   device manager

   \param[in] dev the device to be set as active device
              sycl::device* will need to be cast to void*
   \returns \ref af_err error code
*/
AFAPI af_err afoneapi_set_device(af_sycl_device dev);

/**
   Push user provided device control constructs into the ArrayFire device
   manager pool This function will infer the sycl::device and sycl::context
   corresponding to the provided sycl::queue

   This function should be used only when the user would like ArrayFire to use
   an user generated sycl context and related objects for ArrayFire operations.

   \param[in] que is the user provided sycl queue to be used by ArrayFire
              sycl::queue* will need to be cast to void*
*/
AFAPI af_err afoneapi_add_queue(af_sycl_queue que);

/**
   Remove the user provided device control constructs from the ArrayFire device
   manager pool

   This function should be used only when the user would like ArrayFire to
   remove an already pushed user generated sycl context and related objects.

   \param[in] dev is the sycl device that has to be popped
              sycl::device* will need to be cast to void*

   \note ArrayFire does not take control of releasing the objects passed to it.
   The user needs to release them appropriately.
*/
AFAPI af_err afoneapi_delete_device(af_sycl_device dev);

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

namespace afoneapi {

#if AF_API_VERSION >= 39
/**
    \addtogroup oneapi_mat
    @{
*/

/**
Get a handle to ArrayFire's sycl context

\param[in] retain if true calls clRetainContext prior to returning the context
\returns the current sycl::context being used by ArrayFire

*/
static inline sycl::context getContext() {
  sycl::context ctx;
  af_err err = afoneapi_get_context((void *)&ctx);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to get sycl context from arrayfire");
  return ctx;
}

/**
Get a handle to ArrayFire's sycl command queue

\returns the current sycl::queue being used by ArrayFire

*/
static inline sycl::queue getQueue() {
  sycl::queue q;
  af_err err = afoneapi_get_queue((void *)&q);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to get sycl queue from arrayfire");
  return q;
}

/**
   Get the device ID for ArrayFire's current active device
   \returns the cl_device_id of the current device
*/
static inline sycl::device getDevice() {
  sycl::device dev;
  af_err err = afoneapi_get_device((void *)&dev);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to get sycl device");

  return dev;
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
  af_err err = afoneapi_add_queue((void *)&que);
  if (err != AF_SUCCESS)
    throw af::exception(
        "Failed to push user provided queue/device/context to ArrayFire pool");
}

/**
   Set active device with sycl::device

   \param[in] dev is the sycl device id that is to be set as active device
   inside ArrayFire
*/
static inline void setDevice(sycl::device dev) {
  af_err err = afoneapi_set_device((void *)&dev);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to set device based on sycl::device");
}

/**
   Remove the user provided device from the ArrayFire device
   manager pool

   This function should be used only when the user would like ArrayFire to
   remove an already pushed user generated device.

   \param[in] dev is the sycl device id that has to be popped

   \note ArrayFire does not take control of releasing the objects passed to it.
   The user needs to release them appropriately.
*/
static inline void deleteDevice(sycl::device dev) {
  af_err err = afoneapi_delete_device((void *)&dev);
  if (err != AF_SUCCESS)
    throw af::exception(
        "Failed to remove the requested device from ArrayFire device pool");
}

/**
   Get the type of the current device

   \returns sycl::info::device_type
*/
static inline sycl::info::device_type getDeviceType() {
  sycl::info::device_type res;
  af_err err = afoneapi_get_device_type((void *)&res);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to get sycl device type");
  return res;
}

/**
   Get a the sycl::platform of the current device

   \returns sycl::platform
*/
static inline sycl::platform getPlatform() {
  sycl::platform res;
  af_err err = afoneapi_get_platform((void *)&res);
  if (err != AF_SUCCESS)
    throw af::exception("Failed to get sycl platform");
  return res;
}

/**
Create an af::array object from a sycl::buffer

\param[in] idims the dimensions of the buffer
\param[in] buf the sycl memory object
\returns an array object created from the sycl buffer
 */
template <typename T>
static inline af::array array(af::dim4 idims, sycl::buffer<T> buf) {
  af::dtype type = (af::dtype)af::dtype_traits<T>::af_type;
  const unsigned ndims = (unsigned)idims.ndims();
  const dim_t *dims = idims.get();

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
  return afoneapi::array(af::dim4(dim0), buf);
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
  return afoneapi::array(af::dim4(dim0, dim1), buf);
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
  return afoneapi::array(af::dim4(dim0, dim1, dim2), buf);
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
  return afoneapi::array(af::dim4(dim0, dim1, dim2, dim3), buf);
}

/**
   @}
*/
#endif

} // namespace afoneapi

#endif
