/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cusolverDnManager.hpp>
#include <platform.hpp>
#include <debug_cuda.hpp>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace cusolver {

    const char *errorString(cusolverStatus_t err)
    {
        switch(err) {
        case    CUSOLVER_STATUS_SUCCESS                     :   return "CUSOLVER_STATUS_SUCCESS"                    ;
        case    CUSOLVER_STATUS_NOT_INITIALIZED             :   return "CUSOLVER_STATUS_NOT_INITIALIZED"            ;
        case    CUSOLVER_STATUS_ALLOC_FAILED                :   return "CUSOLVER_STATUS_ALLOC_FAILED"               ;
        case    CUSOLVER_STATUS_INVALID_VALUE               :   return "CUSOLVER_STATUS_INVALID_VALUE"              ;
        case    CUSOLVER_STATUS_ARCH_MISMATCH               :   return "CUSOLVER_STATUS_ARCH_MISMATCH"              ;
        case    CUSOLVER_STATUS_MAPPING_ERROR               :   return "CUSOLVER_STATUS_MAPPING_ERROR"              ;
        case    CUSOLVER_STATUS_EXECUTION_FAILED            :   return "CUSOLVER_STATUS_EXECUTION_FAILED"           ;
        case    CUSOLVER_STATUS_INTERNAL_ERROR              :   return "CUSOLVER_STATUS_INTERNAL_ERROR"             ;
        case    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED   :   return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED"  ;
        case    CUSOLVER_STATUS_NOT_SUPPORTED               :   return "CUSOLVER_STATUS_NOT_SUPPORTED"              ;
        case    CUSOLVER_STATUS_ZERO_PIVOT                  :   return "CUSOLVER_STATUS_ZERO_PIVOT"                 ;
        case    CUSOLVER_STATUS_INVALID_LICENSE             :   return "CUSOLVER_STATUS_INVALID_LICENSE"            ;
        default                                             :   return "UNKNOWN";
        }
    }


//RAII class around the cusolver Handle
    class cusolverDnHandle
    {
        cusolverDnHandle_t handle;
    public:

        cusolverDnHandle()
            : handle(0)
        {
            CUSOLVER_CHECK(cusolverDnCreate(&handle));
        }

        ~cusolverDnHandle()
        {
            cusolverDnDestroy(handle);
        }

        cusolverDnHandle_t get() const
        {
            return handle;
        }
    };

    cusolverDnHandle_t getDnHandle()
    {
        static std::unique_ptr<cusolverDnHandle> handle[cuda::DeviceManager::MAX_DEVICES];

        int id = cuda::getActiveDeviceId();

        if(!handle[id]) {
            handle[id].reset(new cusolverDnHandle());
        }

        // FIXME
        // This is not an ideal case. It's just a hack.
        // The correct way to do is to use
        // CUSOLVER_CHECK(cusolverDnSetStream(cuda::getStream(cuda::getActiveDeviceId())))
        // in the class constructor.
        // However, this is causing a lot of the cusolver functions to fail.
        // The only way to fix them is to use cudaDeviceSynchronize() and cudaStreamSynchronize()
        // all over the place, but even then some calls like getrs in solve_lu
        // continue to fail on any stream other than 0.
        //
        // cuSolver Streams patch:
        // https://gist.github.com/shehzan10/414c3d04a40e7c4a03ed3c2e1b9072e7
        //
        CUDA_CHECK(cudaStreamSynchronize(cuda::getStream(id)));

        return handle[id]->get();
    }

}
