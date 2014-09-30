#pragma once
#include <kernel_headers/meanshift.hpp>
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
using cl::LocalSpaceArg;
using cl::NDRange;
using std::string;

namespace opencl
{

namespace kernel
{

static const dim_type THREADS_X = 16;
static const dim_type THREADS_Y = 16;

template<typename T, bool is_color>
void meanshift(Param out, const Param in, float s_sigma, float c_sigma, uint iter)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static Program            msProgs[DeviceManager::MAX_DEVICES];
        static Kernel           msKernels[DeviceManager::MAX_DEVICES];

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D MAX_CHANNELS="<< (is_color ? 3 : 1);

                    buildProgram(msProgs[device], meanshift_cl, meanshift_cl_len, options.str());

                    msKernels[device] = Kernel(msProgs[device], "meanshift");
                });

        auto meanshiftOp = make_kernel<Buffer, KParam,
                                       Buffer, KParam,
                                       LocalSpaceArg, dim_type,
                                       dim_type, float,
                                       dim_type, float,
                                       unsigned, dim_type
                                      >(msKernels[device]);

        NDRange local(THREADS_X, THREADS_Y);

        dim_type blk_x = divup(in.info.dims[0], THREADS_X);
        dim_type blk_y = divup(in.info.dims[1], THREADS_Y);

        const dim_type bIndex   = (is_color ? 3ll : 2ll);
        const dim_type bCount   = in.info.dims[bIndex];
        const dim_type channels = (is_color ? in.info.dims[2] : 1ll);

        NDRange global(bCount*blk_x*THREADS_X, blk_y*THREADS_Y);

        // clamp spatical and chromatic sigma's
        float space_     = std::min(11.5f, s_sigma);
        dim_type radius  = std::max((dim_type)(space_ * 1.5f), 1ll);
        dim_type padding = 2*radius+1;
        const float cvar = c_sigma*c_sigma;
        size_t loc_size  = channels*(local[0]+padding)*(local[1]+padding)*sizeof(T);

        meanshiftOp(EnqueueArgs(getQueue(), global, local),
                out.data, out.info, in.data, in.info,
                cl::Local(loc_size), bIndex, channels,
                space_, radius, cvar, iter, blk_x);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
