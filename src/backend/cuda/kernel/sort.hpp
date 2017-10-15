/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <math.hpp>
#include <common/dispatch.hpp>
#include <Param.hpp>
#include <kernel/iota.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <thrust/sort.h>
#include <kernel/thrust_sort_by_key.hpp>

namespace cuda
{
    namespace kernel
    {
        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void sort0Iterative(Param<T> val, bool isAscending)
        {
            for(int w = 0; w < val.dims[3]; w++) {
                int valW = w * val.strides[3];
                for(int z = 0; z < val.dims[2]; z++) {
                    int valWZ = valW + z * val.strides[2];
                    for(int y = 0; y < val.dims[1]; y++) {

                        int valOffset = valWZ + y * val.strides[1];

                        if(isAscending) {
                            THRUST_SELECT(thrust::sort,
                                          val.ptr + valOffset,
                                          val.ptr + valOffset + val.dims[0]);
                        } else {
                            THRUST_SELECT(thrust::sort,
                                          val.ptr + valOffset,
                                          val.ptr + valOffset + val.dims[0],
                                          thrust::greater<T>());
                        }
                    }
                }
            }
            POST_LAUNCH_CHECK();
        }

        template<typename T, int dim>
        void sortBatched(Param<T> pVal, bool isAscending)
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
            kernel::iota<uint>(pKey, seqDims, tileDims);

            // Flat
            //val.modDims(inDims.elements());
            //key.modDims(inDims.elements());
            pKey.dims[0] = inDims.elements();
            pKey.strides[0] = 1;
            pVal.dims[0] = inDims.elements();
            pVal.strides[0] = 1;
            for(int i = 1; i < 4; i++) {
                pKey.dims[i] = 1;
                pKey.strides[i] = pKey.strides[i - 1] * pKey.dims[i - 1];
                pVal.dims[i] = 1;
                pVal.strides[i] = pVal.strides[i - 1] * pVal.dims[i - 1];
            }

            // Sort indices
            // sort_by_key<T, uint, isAscending>(*resVal, *resKey, val, key, 0);
            thrustSortByKey(pVal.ptr, pKey.ptr, pVal.dims[0], isAscending);

            // Needs to be ascending (true) in order to maintain the indices properly
            thrustSortByKey(pKey.ptr, pVal.ptr, pVal.dims[0], true);

            // No need of doing moddims here because the original Array<T>
            // dimensions have not been changed
            //val.modDims(inDims);

            // Not really necessary
            // CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
            memFree(key);
        }

        template<typename T>
        void sort0(Param<T> val, bool isAscending)
        {
            int higherDims =  val.dims[1] * val.dims[2] * val.dims[3];

            if(higherDims > 10)
              sortBatched<T, 0>(val, isAscending);
            else
              kernel::sort0Iterative<T>(val, isAscending);
        }
    }
}
