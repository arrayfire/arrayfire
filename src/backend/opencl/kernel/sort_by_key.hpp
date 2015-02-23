/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>
#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/sort_by_key.hpp>
#include <boost/compute/functional/operator.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace compute = boost::compute;

using cl::Buffer;
using cl::Program;
using cl::Kernel;
using cl::make_kernel;
using cl::EnqueueArgs;
using cl::NDRange;
using std::string;

namespace opencl
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const dim_type TX = 32;
        static const dim_type TY = 8;

        template<typename Tk, typename Tv, bool isAscending>
        void sort0_by_key(Param okey, Param oval)
        {
            try {
                compute::command_queue c_queue(getQueue()());

                compute::buffer okey_buf((*okey.data)());
                compute::buffer oval_buf((*oval.data)());

                for(dim_type w = 0; w < okey.info.dims[3]; w++) {
                    dim_type okeyW = w * okey.info.strides[3];
                    dim_type ovalW = w * oval.info.strides[3];
                    for(dim_type z = 0; z < okey.info.dims[2]; z++) {
                        dim_type okeyWZ = okeyW + z * okey.info.strides[2];
                        dim_type ovalWZ = ovalW + z * oval.info.strides[2];
                        for(dim_type y = 0; y < okey.info.dims[1]; y++) {

                            dim_type okeyOffset = okeyWZ + y * okey.info.strides[1];
                            dim_type ovalOffset = ovalWZ + y * oval.info.strides[1];

                            compute::buffer_iterator<Tk> start= compute::make_buffer_iterator<Tk>(okey_buf, okeyOffset);
                            compute::buffer_iterator<Tk> end = compute::make_buffer_iterator<Tk>(okey_buf, okeyOffset + okey.info.dims[0]);
                            compute::buffer_iterator<Tv> vals = compute::make_buffer_iterator<Tv>(oval_buf, ovalOffset);
                            if(isAscending) {
                                compute::sort_by_key(start, end, vals, c_queue);
                            } else {
                                compute::sort_by_key(start, end, vals,
                                                     compute::greater<Tk>(), c_queue);
                            }
                        }
                    }
                }

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}
