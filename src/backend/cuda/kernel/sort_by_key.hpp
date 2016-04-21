/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <kernel/sort_helper.hpp>
#include <kernel/iota.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace cuda
{
    namespace kernel
    {
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

        template<typename Tk, typename Tv, bool isAscending, int dim>
        void sortByKeyBatched(Param<Tk> pKey, Param<Tv> pVal)
        {
            af::dim4 inDims;
            for(int i = 0; i < 4; i++)
                inDims[i] = pKey.dims[i];

            // Sort dimension
            // tileDims * seqDims = inDims
            af::dim4 tileDims(1);
            af::dim4 seqDims = inDims;
            tileDims[dim] = inDims[dim];
            seqDims[dim] = 1;

            // Create/call iota
            // Array<uint> key = iota<uint>(seqDims, tileDims);
            dim4 keydims = inDims;
            uint* key = memAlloc<uint>(keydims.elements());
            Param<uint> pSeq;
            pSeq.ptr = key;
            pSeq.strides[0] = 1;
            pSeq.dims[0] = keydims[0];
            for(int i = 1; i < 4; i++) {
                pSeq.dims[i] = keydims[i];
                pSeq.strides[i] = pSeq.strides[i - 1] * pSeq.dims[i - 1];
            }
            cuda::kernel::iota<uint>(pSeq, seqDims, tileDims);

            // Make pkey, pVal into a pair
            thrust::device_vector<IndexPair<Tk, Tv> > X(inDims.elements());
            IndexPair<Tk, Tv> *Xptr = thrust::raw_pointer_cast(X.data());

            const int threads = 256;
            int blocks = divup(inDims.elements(), threads * copyPairIter);
            CUDA_LAUNCH((makeIndexPair<Tk, Tv>), blocks, threads,
                        Xptr, pKey.ptr, pVal.ptr, inDims.elements());
            POST_LAUNCH_CHECK();

            // Sort indices
            // Need to convert pSeq to thrust::device_ptr, otherwise thrust
            // throws weird errors for all *64 data types (double, intl, uintl etc)
            thrust::device_ptr<uint> dSeq = thrust::device_pointer_cast(pSeq.ptr);
            THRUST_SELECT(thrust::stable_sort_by_key,
                          X.begin(), X.end(),
                          dSeq,
                          IPCompare<Tk, Tv, isAscending>());
            POST_LAUNCH_CHECK();

            // Needs to be ascending (true) in order to maintain the indices properly
            //kernel::sort0_by_key<uint, T, true>(pKey, pVal);
            THRUST_SELECT(thrust::stable_sort_by_key,
                          dSeq,
                          dSeq + inDims.elements(),
                          X.begin());
            POST_LAUNCH_CHECK();

            CUDA_LAUNCH((splitIndexPair<Tk, Tv>), blocks, threads,
                        pKey.ptr, pVal.ptr, Xptr, inDims.elements());
            POST_LAUNCH_CHECK();

            // No need of doing moddims here because the original Array<T>
            // dimensions have not been changed
            //val.modDims(inDims);

            memFree(key);
        }

        template<typename Tk, typename Tv, bool isAscending>
        void sort0ByKey(Param<Tk> okey, Param<Tv> oval)
        {
            int higherDims =  okey.dims[1] * okey.dims[2] * okey.dims[3];
            // TODO Make a better heurisitic
            if(higherDims > 5)
                kernel::sortByKeyBatched<Tk, Tv, isAscending, 0>(okey, oval);
            else
                kernel::sort0ByKeyIterative<Tk, Tv, isAscending>(okey, oval);
        }
    }
}
