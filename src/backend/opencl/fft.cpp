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
#include <fft.hpp>
#include <err_opencl.hpp>
#include <clFFT.h>
#include <string>
#include <cstdio>

using af::dim4;
using std::string;

namespace opencl
{

#define CLFFT_ERROR_CHECK(call) do {            \
    clfftStatus err = (call);                   \
    if (err!=CLFFT_SUCCESS)                     \
        AF_ERROR("clFFT library call failed",   \
                 AF_ERR_INTERNAL);              \
    } while(0);

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
            clfftTeardown();
        }

    private:
        clFFTPlanner() : mAvailSlotIndex(0) {
            clfftInitSetupData(&fftSetup);
            clfftSetup(&fftSetup);
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

    for(int r=0; r<rank; ++r) {
        sprintf(key_str_temp, "%lu:", clLengths[r]);
        key_string.append(std::string(key_str_temp));
    }

    if(istrides!=NULL) {
        for(int r=0; r<rank; ++r) {
            sprintf(key_str_temp, "%lu:", istrides[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, "%lu:", idist);
        key_string.append(std::string(key_str_temp));
    }

    if (ostrides!=NULL) {
        for(int r=0; r<rank; ++r) {
            sprintf(key_str_temp, "%lu:", ostrides[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, "%lu:", odist);
        key_string.append(std::string(key_str_temp));
    }

    sprintf(key_str_temp, "%d:%lu", (int)precision, batch);
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

    clfftStatus res= clfftDestroyPlan(&planner.mHandles[slot_index]);

    if (res==CLFFT_SUCCESS || res==CLFFT_INVALID_PLAN) {
        clfftPlanHandle temp;

        // getContext() returns object of type Context
        // Context() returns the actual cl_context handle
        clfftStatus res = clfftCreateDefaultPlan(&temp, getContext()(), rank, clLengths);

        switch(res) {
            case CLFFT_INVALID_CONTEXT   : AF_ERROR("clFFT: invalid context   ", AF_ERR_INTERNAL);
            case CLFFT_INVALID_PLATFORM  : AF_ERROR("clFFT: invalid platform  ", AF_ERR_INTERNAL);
            case CLFFT_OUT_OF_HOST_MEMORY: AF_ERROR("clFFT: out of host memory", AF_ERR_INTERNAL);
            case CLFFT_OUT_OF_RESOURCES  : AF_ERROR("clFFT: out of resources  ", AF_ERR_INTERNAL);
            case CLFFT_MEM_OBJECT_ALLOCATION_FAILURE:
                                           AF_ERROR("clFFT: mem object allocation failure", AF_ERR_INTERNAL);
            case CLFFT_NOTIMPLEMENTED    : AF_ERROR("clFFt: feature not implemented", AF_ERR_INTERNAL);
            case CLFFT_SUCCESS:
                {
                    res = clfftSetLayout(temp, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
                    res = clfftSetPlanBatchSize(temp, batch);
                    res = clfftSetPlanDistance(temp, idist, odist);
                    res = clfftSetPlanInStride(temp, rank, istrides);
                    res = clfftSetPlanOutStride(temp, rank, ostrides);
                    res = clfftSetPlanPrecision(temp, precision);
                    res = clfftSetResultLocation(temp, CLFFT_INPLACE);

                    // getQueue() returns object of type CommandQueue
                    // CommandQueue() returns the actual cl_command_queue handle
                    res = clfftBakePlan(temp, 1, &(getQueue()()), NULL, NULL);

                    plan = temp;
                    planner.mHandles[slot_index] = temp;
                    planner.mKeys[slot_index] = key_string;
                    planner.mAvailSlotIndex = (slot_index + 1)%clFFTPlanner::MAX_PLAN_CACHE;
                }
                break;
            default: AF_ERROR("clFFT: unkown error", AF_ERR_INTERNAL);
        }
    } else
        AF_ERROR("clFFTDestroyPlan call failed", AF_ERR_INTERNAL);
}

template<typename T> struct Precision;
template<> struct Precision<cfloat > { enum {type = CLFFT_SINGLE}; };
template<> struct Precision<cdouble> { enum {type = CLFFT_DOUBLE}; };

template<typename T, int rank, clfftDirection direction>
void clfft_common(Array<T> &arr)
{
    const dim4 dims    = arr.dims();
    const dim4 strides = arr.strides();
    size_t io_strides[]= {(size_t)strides[0],
                          (size_t)strides[1],
                          (size_t)strides[2],
                          (size_t)strides[3]};

    size_t rank_dims[3] = {(size_t)dims[0], (size_t)dims[1], (size_t)dims[2]};

    clfftPlanHandle plan;

    find_clfft_plan(plan, (clfftDim)rank, rank_dims,
                    io_strides, io_strides[rank],
                    io_strides, io_strides[rank],
                    (clfftPrecision)Precision<T>::type,
                    dims[rank]);

    CLFFT_ERROR_CHECK( clfftEnqueueTransform(plan, direction, 1, &(getQueue()()), 0, NULL, NULL, &((*arr.get())()), NULL, NULL) );
}

template<int rank>
void computePaddedDims(dim4 &pdims, dim_type const * const pad)
{
    if (rank==1) {
        pdims[0] = pad[0];
    } else if (rank==2) {
        pdims[0] = pad[0];
        pdims[1] = pad[1];
    } else if (rank==3) {
        pdims[0] = pad[0];
        pdims[1] = pad[1];
        pdims[2] = pad[2];
    }
}


template<typename T> T zero() { return 0; }

template<> cfloat zero<cfloat>() { return cfloat({{0.0f, 0.0f}}); }

template<> cdouble zero<cdouble>() { return cdouble({{0.0, 0.0}}); }

//(currently) true is in clFFT if length is a power of 2,3,5
inline bool isSupLen(dim_type length)
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

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType> * fft(Array<inType> const &in, double normalize, dim_type const npad, dim_type const * const pad)
{
    ARG_ASSERT(1, (in.isOwner()==true));

    const dim4 dims = in.dims();
    dim4 pdims(1);

    switch(rank) {
        case 1 :
            ARG_ASSERT(1, (rank==1 && isSupLen(dims[0])));
            computePaddedDims<1>(pdims, pad);
            break;
        case 2 :
            ARG_ASSERT(2, (rank==2 && (isSupLen(dims[0]) || isSupLen(dims[1]))));
            computePaddedDims<2>(pdims, pad);
            break;
        case 3 :
            ARG_ASSERT(3, (rank==3 && (isSupLen(dims[0]) || isSupLen(dims[1]) || isSupLen(dims[2]))));
            computePaddedDims<3>(pdims, pad);
            break;
        default: AF_ERROR("invalid rank", AF_ERR_SIZE);
    }

    pdims[rank] = in.dims()[rank];

    Array<outType> *ret = createPaddedArray<inType, outType>(in, (npad>0 ? pdims : in.dims()), zero<outType>(), normalize);

    clfft_common<outType, rank, CLFFT_FORWARD>(*ret);

    return ret;
}

template<typename T, int rank>
Array<T> * ifft(Array<T> const &in, double normalize, dim_type const npad, dim_type const * const pad)
{
    ARG_ASSERT(1, (in.isOwner()==true));

    const dim4 dims = in.dims();
    dim4 pdims(1);

    switch(rank) {
        case 1 :
            ARG_ASSERT(1, (rank==1 && isSupLen(dims[0])));
            computePaddedDims<1>(pdims, pad);
            break;
        case 2 :
            ARG_ASSERT(2, (rank==2 && (isSupLen(dims[0]) || isSupLen(dims[1]))));
            computePaddedDims<2>(pdims, pad);
            break;
        case 3 :
            ARG_ASSERT(3, (rank==3 && (isSupLen(dims[0]) || isSupLen(dims[1]) || isSupLen(dims[2]))));
            computePaddedDims<3>(pdims, pad);
            break;
        default: AF_ERROR("invalid rank", AF_ERR_SIZE);
    }

    pdims[rank] = in.dims()[rank];

    Array<T> *ret = createPaddedArray<T, T>(in, (npad>0 ? pdims : in.dims()), zero<T>(), normalize);

    clfft_common<T, rank, CLFFT_BACKWARD>(*ret);

    return ret;
}

#define INSTANTIATE1(T1, T2)\
    template Array<T2> * fft <T1, T2, 1, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 2, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T2> * fft <T1, T2, 3, true >(const Array<T1> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE1(float  , cfloat )
INSTANTIATE1(double , cdouble)

#define INSTANTIATE2(T)\
    template Array<T> * fft <T, T, 1, false>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * fft <T, T, 2, false>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * fft <T, T, 3, false>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * ifft<T, 1>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * ifft<T, 2>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad); \
    template Array<T> * ifft<T, 3>(const Array<T> &in, double normalize, dim_type const npad, dim_type const * const pad);

INSTANTIATE2(cfloat )
INSTANTIATE2(cdouble)

}
