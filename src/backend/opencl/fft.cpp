/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <fft.hpp>
#include <err_opencl.hpp>
#include <err_clfft.hpp>
#include <clFFT.h>
#include <math.hpp>
#include <string>
#include <cstdio>
#include <memory.hpp>
#include <iostream>
#include <handle.hpp>

using af::dim4;
using std::string;

namespace opencl
{

// clFFTPlanner will do very basic plan caching.
// it looks for required candidate in mHandles array and returns if found one.
// otherwise, it will create a plan and set it at the mAvailSlotIndex and increment
// the slot index variable in ciruclar fashion 0 to MAX_PLAN_CACHE, then back to zero and repeat.
class clFFTPlanner
{
    friend void find_clfft_plan(clfftPlanHandle &plan,
                                clfftDim rank, size_t *clLengths,
                                size_t *istrides, size_t idist,
                                size_t *ostrides, size_t odist,
                                clfftPrecision precision, size_t batch);

    public:
        static clFFTPlanner& getInstance() {
            static clFFTPlanner single_instance;
            return single_instance;
        }

        ~clFFTPlanner() {
            CLFFT_CHECK(clfftTeardown());
        }

    private:
        clFFTPlanner() : mAvailSlotIndex(0) {
            CLFFT_CHECK(clfftInitSetupData(&fftSetup));
            CLFFT_CHECK(clfftSetup(&fftSetup));
            for(int p=0; p<MAX_PLAN_CACHE; ++p)
                mHandles[p] = 0;
        }
        clFFTPlanner(clFFTPlanner const&);
        void operator=(clFFTPlanner const&);

        static const int MAX_PLAN_CACHE = 5;

        int          mAvailSlotIndex;
        string mKeys[MAX_PLAN_CACHE];

        clfftPlanHandle mHandles[MAX_PLAN_CACHE];

        clfftSetupData  fftSetup;
};

void find_clfft_plan(clfftPlanHandle &plan,
                     clfftDim rank, size_t *clLengths,
                     size_t *istrides, size_t idist,
                     size_t *ostrides, size_t odist,
                     clfftPrecision precision, size_t batch)
{
    clFFTPlanner &planner = clFFTPlanner::getInstance();

    // create the key string
    char key_str_temp[64];
    sprintf(key_str_temp, "%d:", rank);

    string key_string(key_str_temp);

    /* WARNING: DO NOT CHANGE sprintf format specifier */
    for(int r=0; r<rank; ++r) {
        sprintf(key_str_temp, SIZE_T_FRMT_SPECIFIER ":", clLengths[r]);
        key_string.append(std::string(key_str_temp));
    }

    if(istrides!=NULL) {
        for(int r=0; r<rank; ++r) {
            sprintf(key_str_temp, SIZE_T_FRMT_SPECIFIER ":", istrides[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, SIZE_T_FRMT_SPECIFIER ":", idist);
        key_string.append(std::string(key_str_temp));
    }

    if (ostrides!=NULL) {
        for(int r=0; r<rank; ++r) {
            sprintf(key_str_temp, SIZE_T_FRMT_SPECIFIER ":", ostrides[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, SIZE_T_FRMT_SPECIFIER ":", odist);
        key_string.append(std::string(key_str_temp));
    }

    sprintf(key_str_temp, "%d:" SIZE_T_FRMT_SPECIFIER, (int)precision, batch);
    key_string.append(std::string(key_str_temp));

    // find the matching plan_index in the array clFFTPlanner::mKeys
    int plan_index = -1;
    for (int i=0; i<clFFTPlanner::MAX_PLAN_CACHE; ++i) {
        if (key_string==planner.mKeys[i]) {
            plan_index = i;
            break;
        }
    }

    // return mHandles[plan_index] if plan_index valid
    if (plan_index!=-1) {
        plan = planner.mHandles[plan_index];
        return;
    }

    // otherwise create a new plan and set it at mAvailSlotIndex
    // and finally set it to output plan variable
    int slot_index = planner.mAvailSlotIndex;

    if (planner.mHandles[slot_index]) {
        CLFFT_CHECK(clfftDestroyPlan(&planner.mHandles[slot_index]));
        planner.mHandles[slot_index] = 0;
    }

    clfftPlanHandle temp;

    // getContext() returns object of type Context
    // Context() returns the actual cl_context handle
    CLFFT_CHECK(clfftCreateDefaultPlan(&temp, getContext()(), rank, clLengths));

    CLFFT_CHECK(clfftSetLayout(temp, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED));
    CLFFT_CHECK(clfftSetPlanBatchSize(temp, batch));
    CLFFT_CHECK(clfftSetPlanDistance(temp, idist, odist));
    CLFFT_CHECK(clfftSetPlanInStride(temp, rank, istrides));
    CLFFT_CHECK(clfftSetPlanOutStride(temp, rank, ostrides));
    CLFFT_CHECK(clfftSetPlanPrecision(temp, precision));
    CLFFT_CHECK(clfftSetResultLocation(temp, CLFFT_INPLACE));
    CLFFT_CHECK(clfftSetPlanScale(temp, CLFFT_BACKWARD, 1.0));

    // getQueue() returns object of type CommandQueue
    // CommandQueue() returns the actual cl_command_queue handle
    CLFFT_CHECK(clfftBakePlan(temp, 1, &(getQueue()()), NULL, NULL));

    plan = temp;
    planner.mHandles[slot_index] = temp;
    planner.mKeys[slot_index] = key_string;
    planner.mAvailSlotIndex = (slot_index + 1)%clFFTPlanner::MAX_PLAN_CACHE;
}

template<typename T> struct Precision;
template<> struct Precision<cfloat > { enum {type = CLFFT_SINGLE}; };
template<> struct Precision<cdouble> { enum {type = CLFFT_DOUBLE}; };

static void computeDims(size_t rdims[4], const dim4 &idims)
{
    for (int i = 0; i < 4; i++) {
        rdims[i] = (size_t)idims[i];
    }
}

//(currently) true is in clFFT if length is a power of 2,3,5
inline bool isSupLen(dim_t length)
{
    while( length > 1 )
    {
        if( length % 2 == 0 )
            length /= 2;
        else if( length % 3 == 0 )
            length /= 3;
        else if( length % 5 == 0 )
            length /= 5;
        else
            return false;
    }
    return true;
}

template<int rank>
void verifySupported(const dim4 dims)
{
    for (int i = 0; i < rank; i++) {
        ARG_ASSERT(1, isSupLen(dims[i]));
    }
}

template<typename T, int rank, bool direction>
void fft_inplace(Array<T> &in)
{
    verifySupported<rank>(in.dims());
    size_t idims[4], istrides[4], iembed[4];

    computeDims(idims   , in.dims());
    computeDims(iembed  , in.getDataDims());
    computeDims(istrides, in.strides());

    clfftPlanHandle plan;

    int batch = 1;
    for (int i = rank; i < 4; i++) {
        batch *= idims[i];
    }

    find_clfft_plan(plan, (clfftDim)rank, idims,
                    istrides, istrides[rank],
                    istrides, istrides[rank],
                    (clfftPrecision)Precision<T>::type,
                    batch);

    cl_mem imem = (*in.get())();
    cl_command_queue queue = getQueue()();

    CLFFT_CHECK(clfftEnqueueTransform(plan,
                                      direction ? CLFFT_FORWARD : CLFFT_BACKWARD,
                                      1, &queue, 0, NULL, NULL,
                                      &imem, &imem, NULL));
}

#define INSTANTIATE(T)                                      \
    template void fft_inplace<T, 1, true >(Array<T> &in);   \
    template void fft_inplace<T, 2, true >(Array<T> &in);   \
    template void fft_inplace<T, 3, true >(Array<T> &in);   \
    template void fft_inplace<T, 1, false>(Array<T> &in);   \
    template void fft_inplace<T, 2, false>(Array<T> &in);   \
    template void fft_inplace<T, 3, false>(Array<T> &in);

    INSTANTIATE(cfloat )
    INSTANTIATE(cdouble)
}
