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
#include <err_cuda.hpp>
#include <cufft.h>
#include <math.hpp>
#include <string>
#include <cstdio>

using af::dim4;
using std::string;

namespace cuda
{

// cuFFTPlanner will do very basic plan caching.
// it looks for required candidate in mHandles array and returns if found one.
// otherwise, it will create a plan and set it at the mAvailSlotIndex and increment
// the slot index variable in ciruclar fashion 0 to MAX_PLAN_CACHE, then back to zero and repeat.
class cuFFTPlanner
{
    friend void find_cufft_plan(cufftHandle &plan, int rank, int *n,
                                int *inembed, int istride, int idist,
                                int *onembed, int ostride, int odist,
                                cufftType type, int batch);

    public:
        static cuFFTPlanner& getInstance() {
            static cuFFTPlanner single_instance;
            return single_instance;
        }

    private:
        cuFFTPlanner() : mAvailSlotIndex(0) {}
        cuFFTPlanner(cuFFTPlanner const&);
        void operator=(cuFFTPlanner const&);

        static const int MAX_PLAN_CACHE = 5;

        int                  mAvailSlotIndex;
        cufftHandle mHandles[MAX_PLAN_CACHE];
        string         mKeys[MAX_PLAN_CACHE];
};

void find_cufft_plan(cufftHandle &plan, int rank, int *n,
                                        int *inembed, int istride, int idist,
                                        int *onembed, int ostride, int odist,
                                        cufftType type, int batch)
{
    cuFFTPlanner &planner = cuFFTPlanner::getInstance();
    // create the key string
    char key_str_temp[64];
    sprintf(key_str_temp, "%d:", rank);

    string key_string(key_str_temp);

    for(int r=0; r<rank; ++r) {
        sprintf(key_str_temp, "%d:", n[r]);
        key_string.append(std::string(key_str_temp));
    }

    if (inembed!=NULL) {
        for(int r=0; r<rank; ++r) {
            sprintf(key_str_temp, "%d:", inembed[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, "%d:%d:", istride, idist);
        key_string.append(std::string(key_str_temp));
    }
    if (onembed!=NULL) {
        for(int r=0; r<rank; ++r) {
            sprintf(key_str_temp, "%d:", onembed[r]);
            key_string.append(std::string(key_str_temp));
        }
        sprintf(key_str_temp, "%d:%d:", ostride, odist);
        key_string.append(std::string(key_str_temp));
    }

    sprintf(key_str_temp, "%d:%d", (int)type, batch);
    key_string.append(std::string(key_str_temp));

    // find the matching plan_index in the array cuFFTPlanner::mKeys
    int plan_index = -1;
    for (int i=0; i<cuFFTPlanner::MAX_PLAN_CACHE; ++i) {
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
    cufftResult res= cufftDestroy(planner.mHandles[slot_index]);
    if (res==CUFFT_SUCCESS || CUFFT_INVALID_PLAN) {
        cufftHandle temp;
        cufftResult res = cufftPlanMany(&temp, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
        switch(res) {
            case CUFFT_ALLOC_FAILED  : AF_ERROR("cuFFTPlanMany: cuFFT GPU resource allocation failed"   , AF_ERR_INTERNAL);
            case CUFFT_INVALID_VALUE : AF_ERROR("cuFFTPlanMany: invalid parameters passed to cuFFT API" , AF_ERR_INTERNAL);
            case CUFFT_INTERNAL_ERROR: AF_ERROR("cuFFTPlanMany: internal driver detected using cuFFT"   , AF_ERR_INTERNAL);
            case CUFFT_SETUP_FAILED  : AF_ERROR("cuFFTPlanMany: cuFFT library initialization failed"    , AF_ERR_INTERNAL);
            case CUFFT_INVALID_SIZE  : AF_ERROR("cuFFTPlanMany: invalid size parameters passed to cuFFT", AF_ERR_INTERNAL);
            default: //CUFFT_SUCCESS
                {
                    plan = temp;
                    planner.mHandles[slot_index] = temp;
                    planner.mKeys[slot_index] = key_string;
                    planner.mAvailSlotIndex = (slot_index + 1)%cuFFTPlanner::MAX_PLAN_CACHE;
                }
                break;
        }
    } else
        AF_ERROR("cuFFTDestroy call failed", AF_ERR_INTERNAL);
}

template<typename T>
struct cufft_transform;

#define CUFFT_FUNC(T, TRANSFORM_TYPE)                               \
    template<>                                                      \
    struct cufft_transform<T>                                       \
    {                                                               \
        enum { type = CUFFT_##TRANSFORM_TYPE };                     \
        cufftResult                                                 \
            operator() (cufftHandle plan, T *in, T *out, int dir) { \
            return cufftExec##TRANSFORM_TYPE(plan, in, out, dir);   \
        }                                                           \
    };

CUFFT_FUNC(cfloat , C2C)
CUFFT_FUNC(cdouble, Z2Z)

template<int rank>
void computeDims(int *rdims, const dim4 &idims)
{
    if (rank==3) {
        rdims[0] = idims[2];
        rdims[1] = idims[1];
        rdims[2] = idims[0];
    } else if(rank==2) {
        rdims[0] = idims[1];
        rdims[1] = idims[0];
    } else {
        rdims[0] = idims[0];
    }
}

template<typename T, int rank, int direction>
void cufft_common(Array<T> &arr)
{
    const dim4 dims    = arr.dims();
    const dim4 strides = arr.strides();

    int rank_dims[3];

    switch(rank) {
        case 1: computeDims<1>(rank_dims, dims); break;
        case 2: computeDims<2>(rank_dims, dims); break;
        case 3: computeDims<3>(rank_dims, dims); break;
    }

    cufftHandle plan;

    find_cufft_plan(plan, rank, rank_dims,
            NULL, strides[0], strides[rank],
            NULL, strides[0], strides[rank],
            (cufftType)cufft_transform<T>::type, dims[rank]);

    cufft_transform<T> transform;

    transform(plan, arr.get(), arr.get(), direction);
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

    Array<outType> ret = padArray<inType, outType>(in, dims, scalar<outType>(0), norm_factor);

    cufft_common<outType, rank, CUFFT_FORWARD>(ret);

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

    Array<T> ret = padArray<T, T>(in, dims, scalar<T>(0), norm_factor);

    cufft_common<T, rank, CUFFT_INVERSE>(ret);

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
