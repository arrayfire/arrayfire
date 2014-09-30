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

        template<typename T>
        void sort0(Param sx, Param ix, const Param in, const bool dir)
        {
            try {
                compute::command_queue c_queue(getQueue()());

                compute::buffer sx_buf(sx.data());
                compute::buffer ix_buf(ix.data());

                compute::vector<unsigned> ix_vec(
                        compute::counting_iterator<int>(0),
                        compute::counting_iterator<int>(in.info.dims[0]),
                        c_queue);

                if(dir) {
                    compute::sort_by_key(
                            compute::make_buffer_iterator<T>(sx_buf, 0),
                            compute::make_buffer_iterator<T>(sx_buf, sx.info.dims[0]),
                            ix_vec.begin(), c_queue);
                } else {
                    compute::sort_by_key(
                            compute::make_buffer_iterator<T>(sx_buf, 0),
                            compute::make_buffer_iterator<T>(sx_buf, sx.info.dims[0]),
                            ix_vec.begin(), compute::greater<T>(), c_queue);
                }

                ix_buf = ix_vec.get_buffer();

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}
