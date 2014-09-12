#pragma once
#include <kernel_headers/tile.hpp>
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
        // Kernel Launch Config Values
        static const dim_type TX = 32;
        static const dim_type TY = 8;
        static const dim_type TILEX = 512;
        static const dim_type TILEY = 32;

        template<typename T>
        void tile(Param out, const Param in)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static Program           tileProgs[DeviceManager::MAX_DEVICES];
                static Kernel          tileKernels[DeviceManager::MAX_DEVICES];

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName();

                    buildProgram(tileProgs[device],
                                 tile_cl,
                                 tile_cl_len,
                                 options.str());

                    tileKernels[device] = Kernel(tileProgs[device], "tile_kernel");
                });

                auto tileOp = make_kernel<Buffer, const Buffer, const KParam, const KParam,
                                          const dim_type, const dim_type> (tileKernels[device]);

                NDRange local(TX, TY, 1);

                dim_type blocksPerMatX = divup(out.info.dims[0], TILEX);
                dim_type blocksPerMatY = divup(out.info.dims[1], TILEY);
                NDRange global(local[0] * blocksPerMatX * out.info.dims[2],
                               local[1] * blocksPerMatY * out.info.dims[3],
                               1);

                tileOp(EnqueueArgs(getQueue(), global, local),
                       out.data, in.data, out.info, in.info, blocksPerMatX, blocksPerMatY);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                SHOW_CL_ERROR(err);
                throw;
            }
        }
    }
}
