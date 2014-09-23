#pragma once
#include <kernel_headers/morph.hpp>
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

static const dim_type CUBE_X    =  8;
static const dim_type CUBE_Y    =  8;
static const dim_type CUBE_Z    =  4;

template<typename T, bool isDilation, dim_type windLen>
void morph(Param         out,
        const Param      in,
        const Param      mask)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static Program            morProgs[DeviceManager::MAX_DEVICES];
        static Kernel           morKernels[DeviceManager::MAX_DEVICES];

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D isDilation="<< isDilation
                            << " -D windLen=" << windLen;

                    buildProgram(morProgs[device], morph_cl, morph_cl_len, options.str());

                    morKernels[device] = Kernel(morProgs[device], "morph");
                });

        auto morphOp = make_kernel<Buffer, KParam,
                                   Buffer, KParam,
                                   Buffer, cl::LocalSpaceArg,
                                   dim_type
                                  >(morKernels[device]);

        NDRange local(THREADS_X, THREADS_Y);

        dim_type blk_x = divup(in.info.dims[0], THREADS_X);
        dim_type blk_y = divup(in.info.dims[1], THREADS_Y);
        // launch batch * blk_x blocks along x dimension
        NDRange global(blk_x * THREADS_X * in.info.dims[2],
                blk_y * THREADS_Y);

        // copy mask/filter to constant memory
        cl_int se_size   = sizeof(T)*windLen*windLen;
        cl::Buffer mBuff = cl::Buffer(getContext(), CL_MEM_READ_ONLY, se_size);
        getQueue().enqueueCopyBuffer(mask.data, mBuff, 0, 0, se_size);

        // calculate shared memory size
        const int halo    = windLen/2;
        const int padding = 2*halo;
        const int locLen  = THREADS_X + padding + 1;
        const int locSize = locLen * (THREADS_Y+padding);

        morphOp(EnqueueArgs(getQueue(), global, local),
                out.data, out.info, in.data, in.info, mBuff,
                cl::Local(locSize*sizeof(T)), blk_x);

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

template<typename T, bool isDilation, dim_type windLen>
void morph3d(Param       out,
        const Param      in,
        const Param      mask)
{
    try {
        static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
        static Program            morProgs[DeviceManager::MAX_DEVICES];
        static Kernel           morKernels[DeviceManager::MAX_DEVICES];

        int device = getActiveDeviceId();

        std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D isDilation="<< isDilation
                            << " -D windLen=" << windLen;

                    buildProgram(morProgs[device], morph_cl, morph_cl_len, options.str());

                    morKernels[device] = Kernel(morProgs[device], "morph3d");
                });

        auto morphOp = make_kernel<Buffer, KParam,
                                   Buffer, KParam,
                                   Buffer, cl::LocalSpaceArg
                                  >(morKernels[device]);

        NDRange local(CUBE_X, CUBE_Y, CUBE_Z);

        dim_type blk_x = divup(in.info.dims[0], CUBE_X);
        dim_type blk_y = divup(in.info.dims[1], CUBE_Y);
        dim_type blk_z = divup(in.info.dims[2], CUBE_Z);
        // launch batch * blk_x blocks along x dimension
        NDRange global(blk_x * CUBE_X,
                blk_y * CUBE_Y,
                blk_z * CUBE_Z);

        // copy mask/filter to constant memory
        cl_int se_size   = sizeof(T)*windLen*windLen*windLen;
        cl::Buffer mBuff = cl::Buffer(getContext(), CL_MEM_READ_ONLY, se_size);
        getQueue().enqueueCopyBuffer(mask.data, mBuff, 0, 0, se_size);

        // calculate shared memory size
        const int halo    = windLen/2;
        const int padding = 2*halo;
        const int locLen  = CUBE_X+padding+1;
        const int locArea = locLen *(CUBE_Y+padding);
        const int locSize = locArea*(CUBE_Z+padding);

        morphOp(EnqueueArgs(getQueue(), global, local),
                out.data, out.info, in.data, in.info, mBuff, cl::Local(locSize*sizeof(T)));

        CL_DEBUG_FINISH(getQueue());
    } catch (cl::Error err) {
        CL_TO_AF_ERROR(err);
        throw;
    }
}

}

}
