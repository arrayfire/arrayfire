/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/DependencyModule.hpp>
#include <cuda.h>
#include <cusparse_v2.h>

namespace cuda {
class cusparseModule {
    common::DependencyModule module;

   public:
    cusparseModule();
    ~cusparseModule() = default;

    MODULE_MEMBER(cusparseCcsc2dense);
    MODULE_MEMBER(cusparseCcsr2dense);
    MODULE_MEMBER(cusparseCdense2csc);
    MODULE_MEMBER(cusparseCdense2csr);
    MODULE_MEMBER(cusparseCgthr);
    MODULE_MEMBER(cusparseCnnz);
    MODULE_MEMBER(cusparseCreateCsr);
    MODULE_MEMBER(cusparseCreateDnMat);
    MODULE_MEMBER(cusparseCreateDnVec);
    MODULE_MEMBER(cusparseCreateIdentityPermutation);
    MODULE_MEMBER(cusparseCreate);
    MODULE_MEMBER(cusparseCreateMatDescr);
    MODULE_MEMBER(cusparseDcsc2dense);
    MODULE_MEMBER(cusparseDcsr2dense);
    MODULE_MEMBER(cusparseDdense2csc);
    MODULE_MEMBER(cusparseDdense2csr);
    MODULE_MEMBER(cusparseDestroyDnMat);
    MODULE_MEMBER(cusparseDestroyDnVec);
    MODULE_MEMBER(cusparseDestroy);
    MODULE_MEMBER(cusparseDestroyMatDescr);
    MODULE_MEMBER(cusparseDestroySpMat);
    MODULE_MEMBER(cusparseDgthr);
    MODULE_MEMBER(cusparseDnnz);
    MODULE_MEMBER(cusparseScsc2dense);
    MODULE_MEMBER(cusparseScsr2dense);
    MODULE_MEMBER(cusparseSdense2csc);
    MODULE_MEMBER(cusparseSdense2csr);
    MODULE_MEMBER(cusparseSetMatIndexBase);
    MODULE_MEMBER(cusparseSetMatType);
    MODULE_MEMBER(cusparseSetStream);
    MODULE_MEMBER(cusparseSgthr);
    MODULE_MEMBER(cusparseSnnz);
    MODULE_MEMBER(cusparseSpMM_bufferSize);
    MODULE_MEMBER(cusparseSpMM);
    MODULE_MEMBER(cusparseSpMV_bufferSize);
    MODULE_MEMBER(cusparseSpMV);
    MODULE_MEMBER(cusparseXcoo2csr);
    MODULE_MEMBER(cusparseXcoosort_bufferSizeExt);
    MODULE_MEMBER(cusparseXcoosortByColumn);
    MODULE_MEMBER(cusparseXcoosortByRow);
    MODULE_MEMBER(cusparseXcsr2coo);
    MODULE_MEMBER(cusparseZcsc2dense);
    MODULE_MEMBER(cusparseZcsr2dense);

#if CUDA_VERSION >= 11000
    MODULE_MEMBER(cusparseXcsrgeam2Nnz);
    MODULE_MEMBER(cusparseCcsrgeam2_bufferSizeExt);
    MODULE_MEMBER(cusparseCcsrgeam2);
    MODULE_MEMBER(cusparseDcsrgeam2_bufferSizeExt);
    MODULE_MEMBER(cusparseDcsrgeam2);
    MODULE_MEMBER(cusparseScsrgeam2_bufferSizeExt);
    MODULE_MEMBER(cusparseScsrgeam2);
    MODULE_MEMBER(cusparseZcsrgeam2_bufferSizeExt);
    MODULE_MEMBER(cusparseZcsrgeam2);
#else
    MODULE_MEMBER(cusparseXcsrgeamNnz);
    MODULE_MEMBER(cusparseCcsrgeam);
    MODULE_MEMBER(cusparseDcsrgeam);
    MODULE_MEMBER(cusparseScsrgeam);
    MODULE_MEMBER(cusparseZcsrgeam);
#endif

    MODULE_MEMBER(cusparseZdense2csc);
    MODULE_MEMBER(cusparseZdense2csr);
    MODULE_MEMBER(cusparseZgthr);
    MODULE_MEMBER(cusparseZnnz);

    spdlog::logger* getLogger() const noexcept;
};

cusparseModule& getCusparsePlugin() noexcept;

}  // namespace cuda
