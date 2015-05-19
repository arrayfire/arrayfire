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
#include <err_cufft.hpp>
#include <cufft.h>
#include <math.hpp>
#include <string>
#include <cstdio>
#include <memory.hpp>

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
    cufftDestroy(planner.mHandles[slot_index]); // We ignore both return values

    cufftHandle temp;
    cufftResult res = cufftPlanMany(&temp, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);

    // If plan creation fails, clean up the memory we hold on to and try again
    if (res != CUFFT_SUCCESS) {
        garbageCollect();
        CUFFT_CHECK(cufftPlanMany(&temp, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch));
    }

    plan = temp;
    planner.mHandles[slot_index] = temp;
    planner.mKeys[slot_index] = key_string;
    planner.mAvailSlotIndex = (slot_index + 1)%cuFFTPlanner::MAX_PLAN_CACHE;
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
void computeDims(int rdims[rank], const dim4 &idims)
{
    for (int i = 0; i < rank; i++) {
        rdims[i] = idims[(rank -1) - i];
    }
}

template<typename T, int rank, bool direction>
void fft_common(Array<T> &out, const Array<T> &in)
{
    const dim4 idims    = in.dims();
    const dim4 istrides = in.strides();
    const dim4 ostrides = out.strides();

    int in_dims[rank];
    int in_embed[rank];
    int out_embed[rank];

    computeDims<rank>(in_dims, idims);
    computeDims<rank>(in_embed, in.getDataDims());
    computeDims<rank>(out_embed, out.getDataDims());

    int batch = 1;
    for (int i = rank; i < 4; i++) {
        batch *= idims[i];
    }

    cufftHandle plan;
    find_cufft_plan(plan, rank, in_dims,
                    in_embed , istrides[0], istrides[rank],
                    out_embed, ostrides[0], ostrides[rank],
                    (cufftType)cufft_transform<T>::type, batch);

    cufft_transform<T> transform;
    CUFFT_CHECK(transform(plan, (T *)in.get(), out.get(), direction ? CUFFT_FORWARD : CUFFT_INVERSE));
}

void computePaddedDims(dim4 &pdims,
                       const dim4 &idims,
                       const dim_t npad,
                       dim_t const * const pad)
{
    for (int i = 0; i < 4; i++) {
        pdims[i] = (i < (int)npad) ? pad[i] : idims[i];
    }
}

template<typename inType, typename outType, int rank, bool isR2C>
Array<outType> fft(Array<inType> const &in, double norm_factor, dim_t const npad, dim_t const * const pad)
{
    ARG_ASSERT(1, (rank>=1 && rank<=3));

    dim4 pdims(1);
    computePaddedDims(pdims, in.dims(), npad, pad);

    Array<outType> ret = padArray<inType, outType>(in, pdims, scalar<outType>(0), norm_factor);
    fft_common<outType, rank, true>(ret, ret);

    return ret;
}

template<typename T, int rank>
Array<T> ifft(Array<T> const &in, double norm_factor, dim_t const npad, dim_t const * const pad)
{
    ARG_ASSERT(1, (rank>=1 && rank<=3));

    dim4 pdims(1);
    computePaddedDims(pdims, in.dims(), npad, pad);

    Array<T> ret = padArray<T, T>(in, pdims, scalar<T>(0), norm_factor);
    fft_common<T, rank, false>(ret, ret);

    return ret;
}

#define INSTANTIATE1(T1, T2)\
    template Array<T2> fft <T1, T2, 1, true >(const Array<T1> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T2> fft <T1, T2, 2, true >(const Array<T1> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T2> fft <T1, T2, 3, true >(const Array<T1> &in, double norm_factor, dim_t const npad, dim_t const * const pad);

INSTANTIATE1(float  , cfloat )
INSTANTIATE1(double , cdouble)

#define INSTANTIATE2(T)\
    template Array<T> fft <T, T, 1, false>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T> fft <T, T, 2, false>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T> fft <T, T, 3, false>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T> ifft<T, 1>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T> ifft<T, 2>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad); \
    template Array<T> ifft<T, 3>(const Array<T> &in, double norm_factor, dim_t const npad, dim_t const * const pad);

INSTANTIATE2(cfloat )
INSTANTIATE2(cdouble)

}
