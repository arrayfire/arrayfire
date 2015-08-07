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
#include <debug_cuda.hpp>
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
    cufftSetStream(plan, cuda::getStream(cuda::getActiveDeviceId()));
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

template<typename To, typename Ti>
struct cufft_real_transform;

#define CUFFT_REAL_FUNC(To, Ti, TRANSFORM_TYPE)                 \
    template<>                                                  \
    struct cufft_real_transform<To, Ti>                         \
    {                                                           \
        enum { type = CUFFT_##TRANSFORM_TYPE };                 \
        cufftResult                                             \
            operator() (cufftHandle plan, Ti *in, To *out) {    \
            return cufftExec##TRANSFORM_TYPE(plan, in, out);    \
        }                                                       \
    };

CUFFT_REAL_FUNC(cfloat , float , R2C)
CUFFT_REAL_FUNC(cdouble, double, D2Z)

CUFFT_REAL_FUNC(float , cfloat , C2R)
CUFFT_REAL_FUNC(double, cdouble, Z2D)

template<int rank>
void computeDims(int rdims[rank], const dim4 &idims)
{
    for (int i = 0; i < rank; i++) {
        rdims[i] = idims[(rank -1) - i];
    }
}

template<typename T, int rank, bool direction>
void fft_inplace(Array<T> &in)
{
    const dim4 idims    = in.dims();
    const dim4 istrides = in.strides();

    int t_dims[rank];
    int in_embed[rank];

    computeDims<rank>(t_dims, idims);
    computeDims<rank>(in_embed, in.getDataDims());

    int batch = 1;
    for (int i = rank; i < 4; i++) {
        batch *= idims[i];
    }

    cufftHandle plan;
    find_cufft_plan(plan, rank, t_dims,
                    in_embed , istrides[0], istrides[rank],
                    in_embed , istrides[0], istrides[rank],
                    (cufftType)cufft_transform<T>::type, batch);

    cufft_transform<T> transform;
    CUFFT_CHECK(transform(plan, (T *)in.get(), in.get(), direction ? CUFFT_FORWARD : CUFFT_INVERSE));
}

template<typename Tc, typename Tr, int rank>
Array<Tc> fft_r2c(const Array<Tr> &in)
{
    dim4 idims = in.dims();
    dim4 odims = in.dims();

    odims[0] = odims[0] / 2 + 1;

    Array<Tc> out = createEmptyArray<Tc>(odims);

    int t_dims[rank];
    int in_embed[rank], out_embed[rank];

    computeDims<rank>(t_dims, idims);
    computeDims<rank>(in_embed, in.getDataDims());
    computeDims<rank>(out_embed, out.getDataDims());

    int batch = 1;
    for (int i = rank; i < 4; i++) {
        batch *= idims[i];
    }

    dim4 istrides = in.strides();
    dim4 ostrides = out.strides();

    cufftHandle plan;
    find_cufft_plan(plan, rank, t_dims,
                    in_embed  , istrides[0], istrides[rank],
                    out_embed , ostrides[0], ostrides[rank],
                    (cufftType)cufft_real_transform<Tc, Tr>::type, batch);

    cufft_real_transform<Tc, Tr> transform;
    CUFFT_CHECK(transform(plan, (Tr *)in.get(), out.get()));
    return out;
}

template<typename Tr, typename Tc, int rank>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims)
{
    Array<Tr> out = createEmptyArray<Tr>(odims);

    int t_dims[rank];
    int in_embed[rank], out_embed[rank];

    computeDims<rank>(t_dims, odims);
    computeDims<rank>(in_embed, in.getDataDims());
    computeDims<rank>(out_embed, out.getDataDims());

    int batch = 1;
    for (int i = rank; i < 4; i++) {
        batch *= odims[i];
    }

    dim4 istrides = in.strides();
    dim4 ostrides = out.strides();

    cufft_real_transform<Tr, Tc> transform;

    cufftHandle plan;
    find_cufft_plan(plan, rank, t_dims,
                    in_embed  , istrides[0], istrides[rank],
                    out_embed , ostrides[0], ostrides[rank],
                    (cufftType)cufft_real_transform<Tr, Tc>::type, batch);

    CUFFT_CHECK(transform(plan, (Tc *)in.get(), out.get()));
    return out;
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

#define INSTANTIATE_REAL(Tr, Tc)                                        \
    template Array<Tc> fft_r2c<Tc, Tr, 1>(const Array<Tr> &in);         \
    template Array<Tc> fft_r2c<Tc, Tr, 2>(const Array<Tr> &in);         \
    template Array<Tc> fft_r2c<Tc, Tr, 3>(const Array<Tr> &in);         \
    template Array<Tr> fft_c2r<Tr, Tc, 1>(const Array<Tc> &in, const dim4 &odims); \
    template Array<Tr> fft_c2r<Tr, Tc, 2>(const Array<Tc> &in, const dim4 &odims); \
    template Array<Tr> fft_c2r<Tr, Tc, 3>(const Array<Tc> &in, const dim4 &odims); \

    INSTANTIATE_REAL(float , cfloat )
    INSTANTIATE_REAL(double, cdouble)
}
