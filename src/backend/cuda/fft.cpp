/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <fft.hpp>
#include <debug_cuda.hpp>
#include <err_cufft.hpp>
#include <cufft.h>
#include <math.hpp>
#include <string>
#include <deque>
#include <utility>
#include <cstdio>
#include <memory.hpp>

using af::dim4;
using std::string;

namespace cuda
{

typedef std::pair<string, cufftHandle> FFTPlanPair;
typedef std::deque<FFTPlanPair> FFTPlanCache;

// cuFFTPlanner caches fft plans
//
// new plan |--> IF number of plans cached is at limit, pop the least used entry and push new plan.
//          |
//          |--> ELSE just push the plan
// existing plan -> reuse a plan
class cuFFTPlanner
{
    friend void find_cufft_plan(cufftHandle &plan, int rank, int *n,
                                int *inembed, int istride, int idist,
                                int *onembed, int ostride, int odist,
                                cufftType type, int batch);

    public:
        static cuFFTPlanner& getInstance() {
            static cuFFTPlanner instances[cuda::DeviceManager::MAX_DEVICES];
            return instances[cuda::getActiveDeviceId()];
        }

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

void find_cufft_plan(cufftHandle &plan, int rank, int *n,
                     int *inembed, int istride, int idist,
                     int *onembed, int ostride, int odist,
                     cufftType type, int batch)
{
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
    cuFFTPlanner &planner = cuFFTPlanner::getInstance();

    int planIndex = planner.findIfPlanExists(key_string);

    // if found a valid plan, return it
    if (planIndex!=-1) {
        plan = planner.getPlan(planIndex);
        return;
    }

    cufftHandle temp;
    cufftResult res = cufftPlanMany(&temp, rank, n,
                                    inembed, istride, idist,
                                    onembed, ostride, odist,
                                    type, batch);

    // If plan creation fails, clean up the memory we hold on to and try again
    if (res != CUFFT_SUCCESS) {
        garbageCollect();
        CUFFT_CHECK(cufftPlanMany(&temp, rank, n,
                                  inembed, istride, idist,
                                  onembed, ostride, odist,
                                  type, batch));
    }

    plan = temp;
    cufftSetStream(plan, cuda::getStream(cuda::getActiveDeviceId()));

    // push the plan into plan cache
    planner.pushPlan(key_string, plan);
}

void setFFTPlanCacheSize(size_t numPlans)
{
    cuFFTPlanner::getInstance().setMaxCacheSize(numPlans);
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
