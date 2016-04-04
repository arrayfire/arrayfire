/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <dispatch.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>

// This needs to be in global namespace as it is used by thrust
template<typename T>
struct IndexPair
{
    T val;
    uint idx;
};

template <typename T, bool isAscending>
struct IPCompare
{
    __host__ __device__
    bool operator()(const IndexPair<T> &lhs, const IndexPair<T> &rhs) const
    {
        // Check stable sort condition
        if(isAscending) return (lhs.val < rhs.val);
        else return (lhs.val > rhs.val);
    }
};

namespace cuda
{
    namespace kernel
    {
        static const int copyPairIter = 4;

        template <typename Tk, typename Tv>
        __global__
        void makeIndexPair(IndexPair<Tv> *out, const Tk *key, const Tv *val, const int N)
        {
            int tIdx = blockIdx.x * blockDim.x * copyPairIter + threadIdx.x;

            for(int i = tIdx; i < N; i += blockDim.x)
            {
                out[i].val = val[i];
                out[i].idx = key[i];
            }
        }

        template <typename Tk, typename Tv>
        __global__
        void splitIndexPair(Tk *key, Tv *val, const IndexPair<Tv> *out, const int N)
        {
            int tIdx = blockIdx.x * blockDim.x * copyPairIter + threadIdx.x;

            for(int i = tIdx; i < N; i += blockDim.x)
            {
                val[i] = out[i].val;
                key[i] = out[i].idx;
            }
        }
    }
}
