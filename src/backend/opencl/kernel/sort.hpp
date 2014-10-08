#pragma once
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

#include <boost/compute.hpp>
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

        template<typename T, bool DIR>
        void sort0(Param sx, Param ix, const Param in)
        {
            try {
                compute::command_queue c_queue(getQueue()());

                compute::buffer sx_buf(sx.data());
                compute::buffer ix_buf(ix.data());

                for(dim_type w = 0; w < in.info.dims[3]; w++) {
                    for(dim_type z = 0; z < in.info.dims[2]; z++) {
                        for(dim_type y = 0; y < in.info.dims[1]; y++) {

                            dim_type sxOffset = w * sx.info.strides[3] + z * sx.info.strides[2]
                                              + y * sx.info.strides[1];
                            dim_type ixOffset = w * ix.info.strides[3] + z * ix.info.strides[2]
                                              + y * ix.info.strides[1];

                            compute::buffer_iterator<unsigned> ix_begin(ix_buf, ixOffset);
                            compute::iota(ix_begin, ix_begin + in.info.dims[0], 0, c_queue);

                            if(DIR) {
                                compute::sort_by_key(
                                        compute::make_buffer_iterator<T>(sx_buf, sxOffset),
                                        compute::make_buffer_iterator<T>(sx_buf, sxOffset + sx.info.dims[0]),
                                        ix_begin, compute::less<T>(), c_queue);
                            } else {
                                compute::sort_by_key(
                                        compute::make_buffer_iterator<T>(sx_buf, sxOffset),
                                        compute::make_buffer_iterator<T>(sx_buf, sxOffset + sx.info.dims[0]),
                                        ix_begin, compute::greater<T>(), c_queue);
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
