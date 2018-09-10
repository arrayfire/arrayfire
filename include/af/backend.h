/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 32
/**
   \param[in] bknd takes one of the values of enum \ref af_backend
   \returns \ref af_err error code

   \ingroup unified_func_setbackend
 */
AFAPI af_err af_set_backend(const af_backend bknd);
#endif

#if AF_API_VERSION >= 32
/**
   \param[out] num_backends Number of available backends
   \returns \ref af_err error code

   \ingroup unified_func_getbackendcount
 */
AFAPI af_err af_get_backend_count(unsigned* num_backends);
#endif

#if AF_API_VERSION >= 32
/**
   Returns a flag of all available backends

   \code{.cpp}
   int backends = 0;
   af_get_available_backends(&backends);

   if(backends & AF_BACKEND_CUDA) {
       // The CUDA backend is available
   }
   \endcode

   \param[out] backends A flag of all available backends. Use the &(and)
   operator to check if a particular backend is available

   \returns \ref af_err error code

   \ingroup unified_func_getavailbackends
 */
AFAPI af_err af_get_available_backends(int* backends);
#endif

#if AF_API_VERSION >= 32
/**
   \param[out] backend takes one of the values of enum \ref af_backend
   \param[in] in is the array who's backend is to be queried
   \returns \ref af_err error code

   \ingroup unified_func_getbackendid
 */
AFAPI af_err af_get_backend_id(af_backend *backend, const af_array in);
#endif

#if AF_API_VERSION >= 33
/**
   \param[out] backend takes one of the values of enum \ref af_backend
   from the backend that is currently set to active
   \returns \ref af_err error code

   \ingroup unified_func_getactivebackend
 */
AFAPI af_err af_get_active_backend(af_backend *backend);
#endif

#if AF_API_VERSION >= 33
/**
   \param[out] device contains the device on which \p in was created.
   \param[in] in is the array who's device is to be queried.
   \returns \ref af_err error code

   \ingroup unified_func_getdeviceid
 */
AFAPI af_err af_get_device_id(int *device, const af_array in);
#endif


#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace af
{
class array;

#if AF_API_VERSION >= 32
/**
   \param[in] bknd takes one of the values of enum \ref af_backend

   \ingroup unified_func_setbackend
 */
AFAPI void setBackend(const Backend bknd);
#endif

#if AF_API_VERSION >= 32
/**
   \returns Number of available backends

   \ingroup unified_func_getbackendcount
 */
AFAPI unsigned getBackendCount();
#endif

#if AF_API_VERSION >= 32
/**
   Returns a flag of all available backends

   \code{.cpp}
   int backends = getAvailableBackends();

   if(backends & AF_BACKEND_CUDA) {
   // The CUDA backend is available
   }
   \endcode

   \returns A flag of available backends

   \ingroup unified_func_getavailbackends
 */
AFAPI int getAvailableBackends();
#endif

#if AF_API_VERSION >= 32
/**
   \param[in] in is the array who's backend is to be queried
   \returns \ref af_backend which is the backend on which the array is created

   \ingroup unified_func_getbackendid
 */
AFAPI af::Backend getBackendId(const array &in);
#endif

#if AF_API_VERSION >= 33
/**
   \returns \ref af_backend which is the backend is currently active

   \ingroup unified_func_getctivebackend
 */
AFAPI af::Backend getActiveBackend();
#endif

#if AF_API_VERSION >= 33
/**
   \param[in] in is the array who's device is to be queried.
   \returns The id of the device on which this array was created.

   \note Device ID can be the same for arrays belonging to different backends.

   \ingroup unified_func_getdeviceid
 */
AFAPI int getDeviceId(const array &in);
#endif

}
#endif
