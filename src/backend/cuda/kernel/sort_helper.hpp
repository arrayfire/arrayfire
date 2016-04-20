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
template<typename Tk, typename Tv>
struct IndexPair
{
    Tk first;
    Tv second;
};

template <typename Tk, typename Tv, bool isAscending>
struct IPCompare
{
    __host__ __device__
    bool operator()(const IndexPair<Tk, Tv> &lhs, const IndexPair<Tk, Tv> &rhs) const
    {
        // Check stable sort condition
        if(isAscending) return (lhs.first < rhs.first);
        else return (lhs.first > rhs.first);
    }
};

namespace cuda
{
    namespace kernel
    {
        static const int copyPairIter = 4;

        template <typename Tk, typename Tv>
        __global__
        void makeIndexPair(IndexPair<Tk, Tv> *out, const Tk *first, const Tv *second, const int N)
        {
            int tIdx = blockIdx.x * blockDim.x * copyPairIter + threadIdx.x;

            for(int i = tIdx; i < N; i += blockDim.x)
            {
                out[i].first = first[i];
                out[i].second = second[i];
            }
        }

        template <typename Tk, typename Tv>
        __global__
        void splitIndexPair(Tk *first, Tv *second, const IndexPair<Tk, Tv> *out, const int N)
        {
            int tIdx = blockIdx.x * blockDim.x * copyPairIter + threadIdx.x;

            for(int i = tIdx; i < N; i += blockDim.x)
            {
                first[i]  = out[i].first;
                second[i] = out[i].second;
            }
        }
    }
}
