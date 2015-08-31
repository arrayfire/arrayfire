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
   Changes the compute backend at run time

   \param[in] bknd takes one of the values of enum \ref af_backend
   \returns \ref af_err error code
 */
AFAPI af_err af_set_backend(const af_backend bknd);

/**
   Gets the number of available backends

   \param[out] num_backends Number of available backends
   \returns \ref af_err error code
 */
AFAPI af_err af_get_backend_count(unsigned* num_backends);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace af
{

/**
   Changes the compute backend at run time

   \param[in] bknd takes one of the values of enum \ref af_backend
 */
AFAPI void setBackend(const Backend bknd);

/**
   Gets the number of available backends

   \returns Number of available backends
 */
AFAPI unsigned getBackendCount();

}
#endif
