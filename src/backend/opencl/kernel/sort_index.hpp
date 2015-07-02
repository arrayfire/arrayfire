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
#include <boost/compute/algorithm/iota.hpp>
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
        template<typename T, bool isAscending>
        void sort0_index(Param val, Param idx)
        {
            try {
                compute::command_queue c_queue(getQueue()());

                compute::buffer val_buf((*val.data)());
                compute::buffer idx_buf((*idx.data)());

                for(int w = 0; w < (int)val.info.dims[3]; w++) {
                    int valW = w * (int)val.info.strides[3];
                    int idxW = w * idx.info.strides[3];
                    for(int z = 0; z < (int)val.info.dims[2]; z++) {
                        int valWZ = valW + z * (int)val.info.strides[2];
                        int idxWZ = idxW + z * idx.info.strides[2];
                        for(int y = 0; y < (int)val.info.dims[1]; y++) {

                            int valOffset = valWZ + y * val.info.strides[1];
                            int idxOffset = idxWZ + y * idx.info.strides[1];

                            compute::buffer_iterator<unsigned> idx_begin(idx_buf, idxOffset);
                            compute::iota(idx_begin, idx_begin + val.info.dims[0], 0, c_queue);

                            if(isAscending) {
                                compute::sort_by_key(
                                        compute::make_buffer_iterator<T>(val_buf, valOffset),
                                        compute::make_buffer_iterator<T>(val_buf, valOffset + val.info.dims[0]),
                                        idx_begin, compute::less<T>(), c_queue);
                            } else {
                                compute::sort_by_key(
                                        compute::make_buffer_iterator<T>(val_buf, valOffset),
                                        compute::make_buffer_iterator<T>(val_buf, valOffset + val.info.dims[0]),
                                        idx_begin, compute::greater<T>(), c_queue);
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
