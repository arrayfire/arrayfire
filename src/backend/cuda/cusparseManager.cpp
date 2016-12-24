/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cusparseManager.hpp>
#include <platform.hpp>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace cusparse {

    const char *errorString(cusparseStatus_t err)
    {
        switch(err) {
        case    CUSPARSE_STATUS_SUCCESS                     :   return "CUSPARSE_STATUS_SUCCESS"                    ;
        case    CUSPARSE_STATUS_NOT_INITIALIZED             :   return "CUSPARSE_STATUS_NOT_INITIALIZED"            ;
        case    CUSPARSE_STATUS_ALLOC_FAILED                :   return "CUSPARSE_STATUS_ALLOC_FAILED"               ;
        case    CUSPARSE_STATUS_INVALID_VALUE               :   return "CUSPARSE_STATUS_INVALID_VALUE"              ;
        case    CUSPARSE_STATUS_ARCH_MISMATCH               :   return "CUSPARSE_STATUS_ARCH_MISMATCH"              ;
        case    CUSPARSE_STATUS_MAPPING_ERROR               :   return "CUSPARSE_STATUS_MAPPING_ERROR"              ;
        case    CUSPARSE_STATUS_EXECUTION_FAILED            :   return "CUSPARSE_STATUS_EXECUTION_FAILED"           ;
        case    CUSPARSE_STATUS_INTERNAL_ERROR              :   return "CUSPARSE_STATUS_INTERNAL_ERROR"             ;
        case    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED   :   return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED"  ;
        case    CUSPARSE_STATUS_ZERO_PIVOT                  :   return "CUSPARSE_STATUS_ZERO_PIVOT"                 ;
        default                                             :   return "UNKNOWN";
        }
    }


//RAII class around the cusparse Handle
    class cusparseHandle
    {
        cusparseHandle_t handle;
    public:

        cusparseHandle()
            : handle(0)
        {
            CUSPARSE_CHECK(cusparseCreate(&handle));
            CUSPARSE_CHECK(cusparseSetStream(handle, cuda::getStream(cuda::getActiveDeviceId())));
        }

        ~cusparseHandle()
        {
            cusparseDestroy(handle);
        }

        cusparseHandle_t get() const
        {
            return handle;
        }
    };

    cusparseHandle_t getHandle()
    {
        static std::unique_ptr<cusparseHandle> handle[cuda::DeviceManager::MAX_DEVICES];

        int id = cuda::getActiveDeviceId();

        if(!handle[id]) {
            handle[id].reset(new cusparseHandle());
        }

        return handle[id]->get();
    }

}
