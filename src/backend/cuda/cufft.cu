/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cufft.hpp>

#include <memory.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace cuda {
const char *_cufftGetResultString(cufftResult res) {
    switch (res) {
        case CUFFT_SUCCESS: return "cuFFT: success";

        case CUFFT_INVALID_PLAN: return "cuFFT: invalid plan handle passed";

        case CUFFT_ALLOC_FAILED: return "cuFFT: resources allocation failed";

        case CUFFT_INVALID_TYPE: return "cuFFT: invalid type (deprecated)";

        case CUFFT_INVALID_VALUE:
            return "cuFFT: invalid parameters passed to cuFFT API";

        case CUFFT_INTERNAL_ERROR:
            return "cuFFT: internal error detected using cuFFT";

        case CUFFT_EXEC_FAILED: return "cuFFT: FFT execution failed";

        case CUFFT_SETUP_FAILED: return "cuFFT: library initialization failed";

        case CUFFT_INVALID_SIZE: return "cuFFT: invalid size parameters passed";

        case CUFFT_UNALIGNED_DATA: return "cuFFT: unaligned data (deprecated)";

        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "cuFFT: call is missing parameters";

        case CUFFT_INVALID_DEVICE:
            return "cuFFT: plan execution different than plan creation";

        case CUFFT_PARSE_ERROR: return "cuFFT: plan parse error";

        case CUFFT_NO_WORKSPACE: return "cuFFT: no workspace provided";

        case CUFFT_NOT_IMPLEMENTED: return "cuFFT: not implemented";

        case CUFFT_LICENSE_ERROR: return "cuFFT: license error";

#if CUDA_VERSION >= 8000
        case CUFFT_NOT_SUPPORTED: return "cuFFT: not supported";
#endif
    }

    return "cuFFT: unknown error";
}

SharedPlan findPlan(int rank, int *n, int *inembed, int istride, int idist,
                    int *onembed, int ostride, int odist, cufftType type,
                    int batch) {
    // create the key string
    char key_str_temp[64];
    sprintf(key_str_temp, "%d:", rank);

    std::string key_string(key_str_temp);

    for (int r = 0; r < rank; ++r) {
        sprintf(key_str_temp, "%d:", n[r]);
        key_string.append(std::string(key_str_temp));
    }

    if (inembed != NULL) {
        for (int r = 0; r < rank; ++r) {
            sprintf(key_str_temp, "%d:", inembed[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, "%d:%d:", istride, idist);
        key_string.append(std::string(key_str_temp));
    }

    if (onembed != NULL) {
        for (int r = 0; r < rank; ++r) {
            sprintf(key_str_temp, "%d:", onembed[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, "%d:%d:", ostride, odist);
        key_string.append(std::string(key_str_temp));
    }

    sprintf(key_str_temp, "%d:%d", (int)type, batch);
    key_string.append(std::string(key_str_temp));

    PlanCache &planner = arrayfire::cuda::fftManager();
    SharedPlan retVal  = planner.find(key_string);

    if (retVal) return retVal;

    PlanType *temp  = (PlanType *)malloc(sizeof(PlanType));
    cufftResult res = cufftPlanMany(temp, rank, n, inembed, istride, idist,
                                    onembed, ostride, odist, type, batch);

    // If plan creation fails, clean up the memory we hold on to and try again
    if (res != CUFFT_SUCCESS) {
        arrayfire::cuda::signalMemoryCleanup();
        CUFFT_CHECK(cufftPlanMany(temp, rank, n, inembed, istride, idist,
                                  onembed, ostride, odist, type, batch));
    }

    retVal.reset(temp, [](PlanType *p) {
        cufftDestroy(*p);
        free(p);
    });
    // push the plan into plan cache
    planner.push(key_string, retVal);

    return retVal;
}
}  // namespace cuda
}  // namespace arrayfire
