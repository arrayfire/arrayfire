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
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cuda
{
    namespace kernel
    {
        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, bool isAscending>
        void sort0_index(Param<T> val, Param<unsigned> idx)
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

                        THRUST_SELECT(thrust::sequence, idx_ptr + idxOffset, idx_ptr + idxOffset + idx.dims[0]);
                        if(isAscending) {
                            THRUST_SELECT(thrust::sort_by_key,
                                    val_ptr + valOffset, val_ptr + valOffset + val.dims[0],
                                    idx_ptr + idxOffset);
                        } else {
                            THRUST_SELECT(thrust::sort_by_key,
                                        val_ptr + valOffset, val_ptr + valOffset + val.dims[0],
                                        idx_ptr + idxOffset, thrust::greater<T>());
                        }
                    }
                }
            }
            POST_LAUNCH_CHECK();
        }
    }
}
