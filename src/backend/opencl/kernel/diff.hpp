#pragma once
#include <kernel_headers/diff.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

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
        static const dim_type TX = 16;
        static const dim_type TY = 16;

        template<typename T, unsigned dim, bool isDiff2>
        void diff(Param out, const Param in, const unsigned indims)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static Program           diffProgs[DeviceManager::MAX_DEVICES];
                static Kernel          diffKernels[DeviceManager::MAX_DEVICES];

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName()
                            << " -D DIM="      << dim
                            << " -D isDiff2="  << isDiff2;

                    buildProgram(diffProgs[device],
                                 diff_cl,
                                 diff_cl_len,
                                 options.str());

                    diffKernels[device] = Kernel(diffProgs[device], "diff_kernel");
                });

                auto diffOp = make_kernel<Buffer, const Buffer, const KParam, const KParam,
                                          const dim_type, const dim_type, const dim_type>
                                          (diffKernels[device]);

                NDRange local(TX, TY, 1);
                if(dim == 0 && indims == 1) {
                    local = NDRange(TX * TY, 1, 1);
                }

                dim_type blocksPerMatX = divup(out.info.dims[0], local[0]);
                dim_type blocksPerMatY = divup(out.info.dims[1], local[1]);
                NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                               local[1] * blocksPerMatY * out.info.dims[3],
                               1);

                const dim_type oElem = out.info.dims[0] * out.info.dims[1]
                                     * out.info.dims[2] * out.info.dims[3];

                diffOp(EnqueueArgs(getQueue(), global, local),
                       out.data, in.data, out.info, in.info, oElem, blocksPerMatX, blocksPerMatY);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}
