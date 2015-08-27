/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/exception.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#if AF_API_VERSION >= 31
/**
   Get the stream for the CUDA device with \p id in ArrayFire context

   \param[out] stream CUDA Stream of device with \p id in ArrayFire context
   \param[in] id ArrayFire device id
   \returns \ref af_err error code

   \ingroup cuda_mat
 */
AFAPI af_err afcu_get_stream(cudaStream_t* stream, int id);
#endif

#if AF_API_VERSION >= 31
/**
   Get the native device id of the CUDA device with \p id in ArrayFire context

   \param[out] nativeid native device id of the CUDA device with \p id in ArrayFire context
   \param[in] id ArrayFire device id
   \returns \ref af_err error code

   \ingroup cuda_mat
 */
AFAPI af_err afcu_get_native_id(int* nativeid, int id);
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace afcu
{

#if AF_API_VERSION >= 31
/**
   Get the stream for the CUDA device with \p id in ArrayFire context

   \param[in] id ArrayFire device id
   \returns cuda stream used by CUDA device

   \ingroup cuda_mat
 */
static inline cudaStream_t getStream(int id)
{
    cudaStream_t retVal;
    af_err err = afcu_get_stream(&retVal, id);
    if (err!=AF_SUCCESS)
        throw af::exception("Failed to get CUDA stream from ArrayFire");
    return retVal;
}
#endif

#if AF_API_VERSION >= 31
/**
   Get the native device id of the CUDA device with \p id in ArrayFire context

   \param[in] id ArrayFire device id
   \returns cuda native id of device

   \ingroup cuda_mat
 */
static inline int getNativeId(int id)
{
    int retVal;
    af_err err = afcu_get_native_id(&retVal, id);
    if (err!=AF_SUCCESS)
        throw af::exception("Failed to get CUDA device native id from ArrayFire");
    return retVal;
}
#endif

}
#endif
