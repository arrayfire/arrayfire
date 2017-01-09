/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <clFFT.h>

#include <memory.hpp>

#include <cstdio>
#include <deque>
#include <string>
#include <utility>

namespace opencl
{
class DeviceManager;
}

namespace clfft
{

typedef std::pair<std::string, clfftPlanHandle> FFTPlanPair;
typedef std::deque<FFTPlanPair> FFTPlanCache;

const char * _clfftGetResultString(clfftStatus st);

void findPlan(clfftPlanHandle &plan,
              clfftLayout iLayout, clfftLayout oLayout,
              clfftDim rank, size_t *clLengths,
              size_t *istrides, size_t idist,
              size_t *ostrides, size_t odist,
              clfftPrecision precision, size_t batch);

// clFFTPlanner caches fft plans
//
// new plan |--> IF number of plans cached is at limit, pop the least used entry and push new plan.
//          |
//          |--> ELSE just push the plan
// existing plan -> reuse a plan
class clFFTPlanner
{
    friend class opencl::DeviceManager;

    friend void findPlan(clfftPlanHandle &plan,
                         clfftLayout iLayout, clfftLayout oLayout,
                         clfftDim rank, size_t *clLengths,
                         size_t *istrides, size_t idist,
                         size_t *ostrides, size_t odist,
                         clfftPrecision precision, size_t batch);

    public:
        clFFTPlanner();
        ~clFFTPlanner();

        inline void setMaxCacheSize(size_t size) {
            mCache.resize(size, FFTPlanPair(std::string(""), 0));
        }

        inline size_t getMaxCacheSize() const {
            return mMaxCacheSize;
        }

        inline clfftPlanHandle getPlan(int index) const {
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
        void popPlan();

        // pushes plan to the front of cache(queue)
        void pushPlan(std::string keyString, clfftPlanHandle plan) {
            if (mCache.size()>mMaxCacheSize) {
                popPlan();
            }
            mCache.push_front(FFTPlanPair(keyString, plan));
        }

    private:
        clFFTPlanner(clFFTPlanner const&);
        void operator=(clFFTPlanner const&);

        clfftSetupData  mFFTSetup;

        size_t       mMaxCacheSize;
        FFTPlanCache mCache;
};

}

#define CLFFT_CHECK(fn) do {                        \
        clfftStatus _clfft_st = fn;                 \
        if (_clfft_st != CLFFT_SUCCESS) {           \
            opencl::garbageCollect();               \
            _clfft_st = (fn);                       \
        }                                           \
        if (_clfft_st != CLFFT_SUCCESS) {           \
            char clfft_st_msg[1024];                \
            snprintf(clfft_st_msg,                  \
                     sizeof(clfft_st_msg),          \
                     "clFFT Error (%d): %s\n",      \
                     (int)(_clfft_st),              \
                     clfft::_clfftGetResultString(  \
                         _clfft_st));               \
                                                    \
            AF_ERROR(clfft_st_msg,                  \
                     AF_ERR_INTERNAL);              \
        }                                           \
    } while(0)
