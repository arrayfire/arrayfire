#pragma once
#include <kernel_headers/transform.hpp>
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

        template<typename T, bool isInverse>
        void transform(Param out, const Param in, const Param tf)
        {
            try {
                static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
                static Program      transformProgs[DeviceManager::MAX_DEVICES];
                static Kernel     transformKernels[DeviceManager::MAX_DEVICES];

                int device = getActiveDeviceId();

                std::call_once( compileFlags[device], [device] () {
                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName()
                            << " -D INVERSE="  << (isInverse ? 1 : 0);

                    buildProgram(transformProgs[device],
                                 transform_cl,
                                 transform_cl_len,
                                 options.str());

                    transformKernels[device] = Kernel(transformProgs[device], "transform_kernel");
                });

                auto transformOp = make_kernel<Buffer, const KParam,
                                         const Buffer, const KParam, const Buffer, const KParam,
                                         const dim_type, const dim_type>
                                         (transformKernels[device]);

                const dim_type nimages = in.info.dims[2];
                // Multiplied in src/backend/transform.cpp
                const dim_type ntransforms = out.info.dims[2] / in.info.dims[2];
                NDRange local(TX, TY, 1);

                NDRange global(local[0] * divup(out.info.dims[0], local[0]) * nimages,
                               local[1] * divup(out.info.dims[1], local[1]) * ntransforms,
                               1);

                transformOp(EnqueueArgs(getQueue(), global, local),
                         out.data, out.info, in.data, in.info, tf.data, tf.info, nimages, ntransforms);

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}
