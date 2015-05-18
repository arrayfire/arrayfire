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
#include <thrust/sort.h>

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 32;
        static const unsigned TY = 8;

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T, bool isAscending>
        void sort0(Param<T> val)
        {
            thrust::device_ptr<T> val_ptr = thrust::device_pointer_cast(val.ptr);

            for(int w = 0; w < val.dims[3]; w++) {
                int valW = w * val.strides[3];
                for(int z = 0; z < val.dims[2]; z++) {
                    int valWZ = valW + z * val.strides[2];
                    for(int y = 0; y < val.dims[1]; y++) {

                        int valOffset = valWZ + y * val.strides[1];

                        if(isAscending) {
                            thrust::sort(val_ptr + valOffset, val_ptr + valOffset + val.dims[0]);
                        } else {
                            thrust::sort(val_ptr + valOffset, val_ptr + valOffset + val.dims[0],
                                         thrust::greater<T>());
                        }
                    }
                }
            }
            POST_LAUNCH_CHECK();
        }
    }
}
