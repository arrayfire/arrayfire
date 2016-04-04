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
        template<typename T, bool isAscending>
        void sort0IndexIterative(Param<T> val, Param<unsigned> idx)
        {
            thrust::device_ptr<T>        val_ptr = thrust::device_pointer_cast(val.ptr);
            thrust::device_ptr<unsigned> idx_ptr = thrust::device_pointer_cast(idx.ptr);

            for(int w = 0; w < val.dims[3]; w++) {
                int valW = w * val.strides[3];
                int idxW = w * idx.strides[3];
                for(int z = 0; z < val.dims[2]; z++) {
                    int valWZ = valW + z * val.strides[2];
                    int idxWZ = idxW + z * idx.strides[2];
                    for(int y = 0; y < val.dims[1]; y++) {

                        int valOffset = valWZ + y * val.strides[1];
                        int idxOffset = idxWZ + y * idx.strides[1];

                        if(isAscending) {
                            THRUST_SELECT(thrust::stable_sort_by_key,
                                    val_ptr + valOffset, val_ptr + valOffset + val.dims[0],
                                    idx_ptr + idxOffset);
                        } else {
                            THRUST_SELECT(thrust::stable_sort_by_key,
                                        val_ptr + valOffset, val_ptr + valOffset + val.dims[0],
                                        idx_ptr + idxOffset, thrust::greater<T>());
                        }
                    }
                }
            }
            POST_LAUNCH_CHECK();
        }

        template<typename T, bool isAscending, int dim>
        void sortIndexBatched(Param<T> pVal, Param<unsigned> pIdx)
        {
            af::dim4 inDims;
            for(int i = 0; i < 4; i++)
                inDims[i] = pVal.dims[i];

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
            Param<uint> pKey;
            pKey.ptr = key;
            pKey.strides[0] = 1;
            pKey.dims[0] = keydims[0];
            for(int i = 1; i < 4; i++) {
                pKey.dims[i] = keydims[i];
                pKey.strides[i] = pKey.strides[i - 1] * pKey.dims[i - 1];
            }
            cuda::kernel::iota<uint>(pKey, seqDims, tileDims);

            // Flat - Not required since inplace and both are continuous
            //val.modDims(inDims.elements());
            //key.modDims(inDims.elements());

            // Make val, idx into a pair
            thrust::device_vector<IndexPair<T> > X(inDims.elements());
            IndexPair<T> *Xptr = thrust::raw_pointer_cast(X.data());

            const int threads = 256;
            int blocks = divup(inDims.elements(), threads * copyPairIter);
            CUDA_LAUNCH((makeIndexPair<unsigned, T>), blocks, threads,
                        Xptr, pIdx.ptr, pVal.ptr, inDims.elements());

            // Sort indices
            // sort_by_key<T, uint, isAscending>(*resVal, *resKey, val, key, 0);
            THRUST_SELECT(thrust::stable_sort_by_key,
                          X.begin(), X.end(),
                          pKey.ptr,
                          IPCompare<T, isAscending>());
            POST_LAUNCH_CHECK();

            // Needs to be ascending (true) in order to maintain the indices properly
            //kernel::sort0_by_key<uint, T, true>(pKey, pVal);
            THRUST_SELECT(thrust::stable_sort_by_key,
                          pKey.ptr,
                          pKey.ptr + inDims.elements(),
                          X.begin());
            POST_LAUNCH_CHECK();

            CUDA_LAUNCH((splitIndexPair<unsigned, T>), blocks, threads,
                        pIdx.ptr, pVal.ptr, Xptr, inDims.elements());
            POST_LAUNCH_CHECK();

            // No need of doing moddims here because the original Array<T>
            // dimensions have not been changed
            //val.modDims(inDims);

            memFree(key);
        }

        template<typename T, bool isAscending>
        void sort0Index(Param<T> val, Param<unsigned> idx)
        {
            int higherDims =  val.dims[1] * val.dims[2] * val.dims[3];
            // TODO Make a better heurisitic
            if(higherDims > 5)
                sortIndexBatched<T, isAscending, 0>(val, idx);
            else
                kernel::sort0IndexIterative<T, isAscending>(val, idx);
        }
    }
}
