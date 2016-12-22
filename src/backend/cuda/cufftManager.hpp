/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <err_common.hpp>
#include <cufft.h>

#include <cstdio>
#include <deque>
#include <string>
#include <utility>

namespace cuda
{
class DeviceManager;
}

namespace cufft
{

typedef std::pair<std::string, cufftHandle> FFTPlanPair;
typedef std::deque<FFTPlanPair> FFTPlanCache;

const char * _cufftGetResultString(cufftResult res);

void findPlan(cufftHandle &plan, int rank, int *n,
              int *inembed, int istride, int idist,
              int *onembed, int ostride, int odist,
              cufftType type, int batch);

// cuFFTPlanner caches fft plans
//
// new plan |--> IF number of plans cached is at limit, pop the least used entry and push new plan.
//          |
//          |--> ELSE just push the plan
// existing plan -> reuse a plan
class cuFFTPlanner
{
    friend class cuda::DeviceManager;

    friend void findPlan(cufftHandle &plan, int rank, int *n,
                         int *inembed, int istride, int idist,
                         int *onembed, int ostride, int odist,
                         cufftType type, int batch);

    public:
        inline void setMaxCacheSize(size_t size) {
            mCache.resize(size, FFTPlanPair(std::string(""), 0));
        }

        inline size_t getMaxCacheSize() const {
            return mMaxCacheSize;
        }

        inline cufftHandle getPlan(int index) const {
            return mCache[index].second;
        }

        // iterates through plan cache from front to back
        // of the cache(queue)
        int findIfPlanExists(std::string keyString) const {
            int retVal = -1;
            for(uint i=0; i<mCache.size(); ++i) {
                if (keyString == mCache[i].first) {
                    retVal = i;
                }
            }
            return retVal;
        }

        // pops plan from the back of cache(queue)
        void popPlan() {
            if (!mCache.empty()) {
                // destroy the cufft plan associated with the
                // least recently used plan
                cufftDestroy(mCache.back().second);
                // now pop the entry from cache
                mCache.pop_back();
            }
        }

        // pushes plan to the front of cache(queue)
        void pushPlan(std::string keyString, cufftHandle plan) {
            if (mCache.size()>mMaxCacheSize) {
                popPlan();
            }
            mCache.push_front(FFTPlanPair(keyString, plan));
        }

    private:
        cuFFTPlanner() : mMaxCacheSize(5) {}
        cuFFTPlanner(cuFFTPlanner const&);
        void operator=(cuFFTPlanner const&);

        size_t       mMaxCacheSize;
        FFTPlanCache mCache;
};

}

#define CUFFT_CHECK(fn) do {                        \
        cufftResult _cufft_res = fn;                \
        if (_cufft_res != CUFFT_SUCCESS) {          \
            char cufft_res_msg[1024];               \
            snprintf(cufft_res_msg,                 \
                     sizeof(cufft_res_msg),         \
                     "cuFFT Error (%d): %s\n",      \
                     (int)(_cufft_res),             \
                     cufft::_cufftGetResultString(  \
                         _cufft_res));              \
                                                    \
            AF_ERROR(cufft_res_msg,                 \
                     AF_ERR_INTERNAL);              \
        }                                           \
    } while(0)
