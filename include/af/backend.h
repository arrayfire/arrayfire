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

/**
   \param[in] bknd takes one of the values of enum \ref af_backend
   \returns \ref af_err error code

   \ingroup unified_func_setbackend
 */
AFAPI af_err af_set_backend(const af_backend bknd);

/**
   \param[out] num_backends Number of available backends
   \returns \ref af_err error code

   \ingroup unified_func_getbackendcount
 */
AFAPI af_err af_get_backend_count(unsigned* num_backends);

/**
   \param[out] backends is the OR sum of the backends available.
   \returns \ref af_err error code

   \ingroup unified_func_getavailbackends
 */
AFAPI af_err af_get_available_backends(int* backends);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace af
{

/**
   \param[in] bknd takes one of the values of enum \ref af_backend

   \ingroup unified_func_setbackend
 */
AFAPI void setBackend(const Backend bknd);

/**
   \returns Number of available backends

   \ingroup unified_func_getbackendcount
 */
AFAPI unsigned getBackendCount();

/**
   \returns OR sum of the backends available

   \ingroup unified_func_getavailbackends
 */
AFAPI int getAvailableBackends();

}
#endif
