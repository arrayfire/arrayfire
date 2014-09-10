#pragma once

#include <af/defines.h>
#include <kernel_headers/transform.hpp>
#include <cl.hpp>
#include <platform.hpp>
#include <traits.hpp>
#include <helper.hpp>
#include <sstream>
#include <string>
#include <mutex>
#include <dispatch.hpp>

typedef struct
{
    dim_type dim[4];
} dims_t;

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
        void transform(Buffer out, const Buffer in, const Buffer tf,
              const dim_type *odims, const dim_type *idims,
              const dim_type *ostrides, const dim_type *istrides, const
              dim_type *tstrides, const dim_type i_offset)
        {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static Program            trnProgs[DeviceManager::MAX_DEVICES];
            static Kernel           trnKernels[DeviceManager::MAX_DEVICES];

            int device = getActiveDeviceId();

            std::call_once( compileFlags[device], [device] () {
                    Program::Sources setSrc;
                    setSrc.emplace_back(transform_cl, transform_cl_len);

                    trnProgs[device] = Program(getContext(), setSrc);

                    std::ostringstream options;
                    options << " -D T="        << dtype_traits<T>::getName()
                            << " -D dim_type=" << dtype_traits<dim_type>::getName()
                            << " -D INVERSE="  << (isInverse ? 1 : 0);
                    trnProgs[device].build(options.str().c_str());

                    trnKernels[device] = Kernel(trnProgs[device], "transform_kernel");
                    });

            auto transformOp = make_kernel<Buffer, const dim_type, const dim_type,
                                     const Buffer, const dim_type, const dim_type,
                                     const Buffer, const dims_t, const dims_t,
                                     const dim_type, const dim_type, const dim_type>
                                     (trnKernels[device]);

            const dim_type nimages = idims[2];
            // Multiplied in src/backend/transform.cpp
            const dim_type ntransforms = odims[2] / idims[2];
            NDRange local(TX, TY, 1);

            NDRange global(local[0] * divup(odims[0], local[0]) * nimages,
                           local[1] * divup(odims[1], local[1]) * ntransforms,
                           1);

            dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
            dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};

            transformOp(EnqueueArgs(getQueue(), global, local),
                     out, odims[0], odims[1], in, idims[0], idims[1],
                     tf, _ostrides, _istrides, nimages, ntransforms, i_offset);
        }
    }
}
