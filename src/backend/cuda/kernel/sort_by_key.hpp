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
        template<typename Tk, typename Tv, bool isAscending>
        void sort0_by_key(Param<Tk> okey, Param<Tv> oval)
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
                            THRUST_SELECT(thrust::sort_by_key, okey_ptr + okeyOffset, okey_ptr + okeyOffset + okey.dims[0], oval_ptr + ovalOffset);
                        } else {
                            THRUST_SELECT(thrust::sort_by_key, okey_ptr + okeyOffset, okey_ptr + okeyOffset + okey.dims[0], oval_ptr + ovalOffset, thrust::greater<Tk>());
                        }
                    }
                }
            }
            POST_LAUNCH_CHECK();
        }
    }
}
