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
   \param[out] backends is the OR sum of the backends available.
   \returns \ref af_err error code

   \ingroup unified_func_getavailbackends
 */
AFAPI af_err af_get_available_backends(int* backends);
#endif

/**
   \param[out] backend takes one of the values of enum \ref af_backend
   \param[in] in is the array who's backend is to be queried
   \returns \ref af_err error code

   \ingroup unified_func_getbackendid
 */
AFAPI af_err af_get_backend_id(af_backend *backend, const af_array in);

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
   \returns OR sum of the backends available

   \ingroup unified_func_getavailbackends
 */
AFAPI int getAvailableBackends();
#endif

/**
   \param[in] in is the array who's backend is to be queried
   \returns \ref af_backend which is the backend on which the array is created

   \ingroup unified_func_getbackendid
 */
AFAPI af::Backend getBackendId(const array &in);

}
#endif
