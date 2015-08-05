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

AFAPI af_err afcu_get_stream(cudaStream_t* stream, int id);

AFAPI af_err afcu_get_native_id(int* nativeid, int id);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace afcu
{

static inline cudaStream_t getStream(int id)
{
    cudaStream_t retVal;
    af_err err = afcu_get_stream(&retVal, id);
    if (err!=AF_SUCCESS)
        throw af::exception("Failed to get CUDA stream from ArrayFire");
    return retVal;
}

static inline int getNativeId(int id)
{
    int retVal;
    af_err err = afcu_get_native_id(&retVal, id);
    if (err!=AF_SUCCESS)
        throw af::exception("Failed to get CUDA device native id from ArrayFire");
    return retVal;
}

}
#endif
