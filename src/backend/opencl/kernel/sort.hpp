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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/stable_sort.hpp>
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
        using std::conditional;
        using std::is_same;
        template<typename T>
        using ltype_t = typename conditional<is_same<T, intl>::value, long, T>::type;

        template<typename T>
        using type_t = typename conditional<is_same<T, uintl>::value,
                                            unsigned long, ltype_t<T>
                                           >::type;

        template<typename T, bool isAscending>
        void sort0(Param val)
        {
            try {
                compute::command_queue c_queue(getQueue()());

                compute::buffer val_buf((*val.data)());

                for(int w = 0; w < val.info.dims[3]; w++) {
                    int valW = w * val.info.strides[3];
                    for(int z = 0; z < val.info.dims[2]; z++) {
                        int valWZ = valW + z * val.info.strides[2];
                        for(int y = 0; y < val.info.dims[1]; y++) {

                            int valOffset = valWZ + y * val.info.strides[1];

                            if(isAscending) {
                                compute::stable_sort(
                                        compute::make_buffer_iterator< type_t<T> >(val_buf, valOffset),
                                        compute::make_buffer_iterator< type_t<T> >(val_buf, valOffset + val.info.dims[0]),
                                        compute::less< type_t<T> >(), c_queue);
                            } else {
                                compute::stable_sort(
                                        compute::make_buffer_iterator< type_t<T> >(val_buf, valOffset),
                                        compute::make_buffer_iterator< type_t<T> >(val_buf, valOffset + val.info.dims[0]),
                                        compute::greater< type_t<T> >(), c_queue);
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

#pragma GCC diagnostic pop
