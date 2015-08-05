/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <stdio.h>
#include <platform.hpp>
#include <err_common.hpp>
#include <cublasManager.hpp>
#include <boost/scoped_ptr.hpp>
#include <platform.hpp>

namespace cublas {

    const char *errorString(cublasStatus_t err)
    {

        switch(err)
        {
        case    CUBLAS_STATUS_SUCCESS:              return "CUBLAS_STATUS_SUCCESS";
        case    CUBLAS_STATUS_NOT_INITIALIZED:      return "CUBLAS_STATUS_NOT_INITIALIZED";
        case    CUBLAS_STATUS_ALLOC_FAILED:         return "CUBLAS_STATUS_ALLOC_FAILED";
        case    CUBLAS_STATUS_INVALID_VALUE:        return "CUBLAS_STATUS_INVALID_VALUE";
        case    CUBLAS_STATUS_ARCH_MISMATCH:        return "CUBLAS_STATUS_ARCH_MISMATCH";
        case    CUBLAS_STATUS_MAPPING_ERROR:        return "CUBLAS_STATUS_MAPPING_ERROR";
        case    CUBLAS_STATUS_EXECUTION_FAILED:     return "CUBLAS_STATUS_EXECUTION_FAILED";
        case    CUBLAS_STATUS_INTERNAL_ERROR:       return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION > 5050
        case    CUBLAS_STATUS_NOT_SUPPORTED:        return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
        default:                                    return "UNKNOWN";
        }
    }

    //RAII class around the cublas Handle
    class cublasHandle
    {
        cublasHandle_t handle;
    public:

        cublasHandle()  : handle(0)
        {
            CUBLAS_CHECK(cublasCreate(&handle));
            CUBLAS_CHECK(cublasSetStream(handle, cuda::getStream(cuda::getActiveDeviceId())));
        }

        ~cublasHandle()
        {
            cublasDestroy(handle);
        }

        cublasHandle_t get() const
        {
            return handle;
        }
    };

    cublasHandle_t getHandle()
    {
        using boost::scoped_ptr;
        static scoped_ptr<cublasHandle> handle[cuda::DeviceManager::MAX_DEVICES];

        int id = cuda::getActiveDeviceId();

        if(!handle[id]) {
            handle[id].reset(new cublasHandle());
        }

        return handle[id]->get();
    }
}
