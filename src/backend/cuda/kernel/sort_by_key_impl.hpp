/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <memory.hpp>
#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <kernel/sort_by_key.hpp>
#include <kernel/iota.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

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

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename Tk, typename Tv, bool isAscending>
        void sort0ByKeyIterative(Param<Tk> okey, Param<Tv> oval)
        {
            thrust::device_ptr<Tk>       okey_ptr = thrust::device_pointer_cast(okey.ptr);
            thrust::device_ptr<Tv>       oval_ptr = thrust::device_pointer_cast(oval.ptr);

            for(int w = 0; w < okey.dims[3]; w++) {
                int okeyW = w * okey.strides[3];
                int ovalW = w * oval.strides[3];
                for(int z = 0; z < okey.dims[2]; z++) {
                    int okeyWZ = okeyW + z * okey.strides[2];
                    int ovalWZ = ovalW + z * oval.strides[2];
                    for(int y = 0; y < okey.dims[1]; y++) {

                        int okeyOffset = okeyWZ + y * okey.strides[1];
                        int ovalOffset = ovalWZ + y * oval.strides[1];

                        if(isAscending) {
                            THRUST_SELECT(thrust::stable_sort_by_key,
                                          okey_ptr + okeyOffset,
                                          okey_ptr + okeyOffset + okey.dims[0],
                                          oval_ptr + ovalOffset);
                        } else {
                            THRUST_SELECT(thrust::stable_sort_by_key,
                                          okey_ptr + okeyOffset,
                                          okey_ptr + okeyOffset + okey.dims[0],
                                          oval_ptr + ovalOffset, thrust::greater<Tk>());
                        }
                    }
                }
            }
            POST_LAUNCH_CHECK();
        }

        template<typename Tk, typename Tv, bool isAscending>
        void sortByKeyBatched(Param<Tk> pKey, Param<Tv> pVal, const int dim)
        {
            af::dim4 inDims;
            for(int i = 0; i < 4; i++)
                inDims[i] = pKey.dims[i];

            const dim_t elements = inDims.elements();

            // Sort dimension
            // tileDims * seqDims = inDims
            af::dim4 tileDims(1);
            af::dim4 seqDims = inDims;
            tileDims[dim] = inDims[dim];
            seqDims[dim] = 1;

            // Create/call iota
            // Array<uint> key = iota<uint>(seqDims, tileDims);
            uint* key = memAlloc<uint>(elements);
            Param<uint> pSeq;
            pSeq.ptr = key;
            pSeq.strides[0] = 1;
            pSeq.dims[0] = inDims[0];
            for(int i = 1; i < 4; i++) {
                pSeq.dims[i] = inDims[i];
                pSeq.strides[i] = pSeq.strides[i - 1] * pSeq.dims[i - 1];
            }
            cuda::kernel::iota<uint>(pSeq, seqDims, tileDims);

            // Make pkey, pVal into a pair
            IndexPair<Tk, Tv> *Xptr = (IndexPair<Tk, Tv>*)memAlloc<char>(sizeof(IndexPair<Tk, Tv>) * elements);

            const int threads = 256;
            int blocks = divup(elements, threads * copyPairIter);
            CUDA_LAUNCH((makeIndexPair<Tk, Tv>), blocks, threads,
                        Xptr, pKey.ptr, pVal.ptr, elements);
            POST_LAUNCH_CHECK();

            thrust::device_ptr<IndexPair<Tk, Tv> > X = thrust::device_pointer_cast(Xptr);

            // Sort indices
            // Need to convert pSeq to thrust::device_ptr, otherwise thrust
            // throws weird errors for all *64 data types (double, intl, uintl etc)
            thrust::device_ptr<uint> dSeq = thrust::device_pointer_cast(pSeq.ptr);
            THRUST_SELECT(thrust::stable_sort_by_key,
                          X, X + elements,
                          dSeq,
                          IPCompare<Tk, Tv, isAscending>());
            POST_LAUNCH_CHECK();

            // Needs to be ascending (true) in order to maintain the indices properly
            //kernel::sort0_by_key<uint, T, true>(pKey, pVal);
            THRUST_SELECT(thrust::stable_sort_by_key,
                          dSeq, dSeq + elements,
                          X);
            POST_LAUNCH_CHECK();

            CUDA_LAUNCH((splitIndexPair<Tk, Tv>), blocks, threads,
                        pKey.ptr, pVal.ptr, Xptr, elements);
            POST_LAUNCH_CHECK();

            // No need of doing moddims here because the original Array<T>
            // dimensions have not been changed
            //val.modDims(inDims);

            memFree(key);
            memFree((char*)Xptr);
        }

        template<typename Tk, typename Tv, bool isAscending>
        void sort0ByKey(Param<Tk> okey, Param<Tv> oval)
        {
            int higherDims =  okey.dims[1] * okey.dims[2] * okey.dims[3];
            // TODO Make a better heurisitic
            if(higherDims > 4)
                kernel::sortByKeyBatched<Tk, Tv, isAscending>(okey, oval, 0);
            else
                kernel::sort0ByKeyIterative<Tk, Tv, isAscending>(okey, oval);
        }

#define INSTANTIATE(Tk, Tv, dr)                                                                 \
    template void sort0ByKey<Tk, Tv, dr>(Param<Tk> okey, Param<Tv> oval);                       \
    template void sort0ByKeyIterative<Tk, Tv, dr>(Param<Tk> okey, Param<Tv> oval);              \
    template void sortByKeyBatched<Tk, Tv, dr>(Param<Tk> okey, Param<Tv> oval, const int dim);

#define INSTANTIATE0(Tk    , dr) \
    INSTANTIATE(Tk, float  , dr) \
    INSTANTIATE(Tk, double , dr) \
    INSTANTIATE(Tk, cfloat , dr) \
    INSTANTIATE(Tk, cdouble, dr) \
    INSTANTIATE(Tk, char   , dr) \
    INSTANTIATE(Tk, uchar  , dr)

#define INSTANTIATE1(Tk    , dr) \
    INSTANTIATE(Tk, int    , dr) \
    INSTANTIATE(Tk, uint   , dr) \
    INSTANTIATE(Tk, short  , dr) \
    INSTANTIATE(Tk, ushort , dr) \
    INSTANTIATE(Tk, intl   , dr) \
    INSTANTIATE(Tk, uintl  , dr)

    }
}
