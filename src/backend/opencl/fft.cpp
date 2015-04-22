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
        sprintf(key_str_temp, "%zu:", clLengths[r]);
        key_string.append(std::string(key_str_temp));
    }

    if(istrides!=NULL) {
        for(int r=0; r<rank; ++r) {
            sprintf(key_str_temp, "%zu:", istrides[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, "%zu:", idist);
        key_string.append(std::string(key_str_temp));
    }

    if (ostrides!=NULL) {
        for(int r=0; r<rank; ++r) {
            sprintf(key_str_temp, "%zu:", ostrides[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, "%zu:", odist);
        key_string.append(std::string(key_str_temp));
    }

    sprintf(key_str_temp, "%d:%zu", (int)precision, batch);
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

    CLFFT_CHECK( clfftEnqueueTransform(plan, direction, 1, &(getQueue()()), 0, NULL, NULL, &((*arr.get())()), NULL, NULL) );
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

template<int rank>
void verifySupported(const dim4 dims)
{
    for (int i = 0; i < rank; i++) {
        ARG_ASSERT(1, isSupLen(dims[i]));
    }
}

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType> fft(Array<inType> const &in, double norm_factor, dim_type const npad, dim_type const * const pad)
{
    ARG_ASSERT(1, (in.isOwner()==true));
    ARG_ASSERT(1, (rank>=1 && rank<=3));

    dim4 dims = in.dims();

    dim4 pdims(1);
    computePaddedDims<rank>(pdims, pad);
    pdims[rank] = in.dims()[rank];

    if (npad>0)
      dims = pdims;

    verifySupported<rank>(dims);

    Array<outType> ret = padArray<inType, outType>(in, dims, scalar<outType>(0), norm_factor);
    clfft_common<outType, rank, CLFFT_FORWARD>(ret);

    return ret;
}

template<typename T, int rank>
Array<T> ifft(Array<T> const &in, double norm_factor, dim_type const npad, dim_type const * const pad)
{
    ARG_ASSERT(1, (in.isOwner()==true));
    ARG_ASSERT(1, (rank>=1 && rank<=3));

    dim4 dims = in.dims();

    dim4 pdims(1);
    computePaddedDims<rank>(pdims, pad);
    pdims[rank] = in.dims()[rank];

    if (npad>0)
      dims = pdims;

    // the input norm_factor is further scaled
    // based on the input dimensions to match
    // cuFFT behavior
    for (int i=0; i<rank; i++)
        norm_factor *= dims[i];

    verifySupported<rank>(dims);

    Array<T> ret = padArray<T, T>(in, dims, scalar<T>(0), norm_factor);
    clfft_common<T, rank, CLFFT_BACKWARD>(ret);
    return ret;
}

#define INSTANTIATE1(T1, T2)\
    template Array<T2> fft <T1, T2, 1, true >(const Array<T1> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T2> fft <T1, T2, 2, true >(const Array<T1> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T2> fft <T1, T2, 3, true >(const Array<T1> &in, double norm_factor, dim_type const npad, dim_type const * const pad);

INSTANTIATE1(float  , cfloat )
INSTANTIATE1(double , cdouble)

#define INSTANTIATE2(T)\
    template Array<T> fft <T, T, 1, false>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T> fft <T, T, 2, false>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T> fft <T, T, 3, false>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T> ifft<T, 1>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T> ifft<T, 2>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad); \
    template Array<T> ifft<T, 3>(const Array<T> &in, double norm_factor, dim_type const npad, dim_type const * const pad);

INSTANTIATE2(cfloat )
INSTANTIATE2(cdouble)

}
