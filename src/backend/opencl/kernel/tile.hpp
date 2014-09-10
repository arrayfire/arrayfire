#pragma once

#include <af/defines.h>
#include <kernel_headers/tile.hpp>
#include <cl.hpp>
#include <ctx.hpp>
#include <traits.hpp>
#include <helper.hpp>
#include <sstream>
#include <string>
#include <dispatch.hpp>

typedef struct
{
    dim_type dims[4];
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
        // Kernel Launch Config Values
        static const dim_type TX = 32;
        static const dim_type TY = 8;
        static const dim_type TILEX = 512;
        static const dim_type TILEY = 32;

        template<typename T>
        void tile(Buffer out, const Buffer in,
                  const dim_type *odims, const dim_type *idims,
                  const dim_type *ostrides, const dim_type *istrides, const dim_type offset)
        {

            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static Program            tileProgs[DeviceManager::MAX_DEVICES];
            static Kernel           tileKernels[DeviceManager::MAX_DEVICES];

            int device = getActiveDeviceId();

            std::call_once( compileFlags[device], [device] () {
                    Program::Sources setSrc;
                    setSrc.emplace_back(transpose_cl, transpose_cl_len);

                    tileProgs[device] = Program(getContext(), setSrc);

                    std::ostringstream options;
                    options << " -D T=" << dtype_traits<T>::getName()
                            << " -D dim_type=" << dtype_traits<dim_type>::getName()
                            << " -D TILE_DIM=" << TILE_DIM;
                    tileProgs[device].build(options.str().c_str());

                    tileKernels[device] = Kernel(tileProgs[device], "transpose");
                });

            auto tileOp = make_kernel<Buffer, const Buffer, const dims_t, const dims_t,
                                      const dims_t, const dims_t, const dim_type,
                                      const unsigned, const unsigned>
                                      (tileKernels[device]);

            NDRange local(TX, TY, 1);

            unsigned blocksPerMatX = divup(odims[0], TILEX);
            unsigned blocksPerMatY = divup(odims[1], TILEY);
            NDRange global(local[0] * blocksPerMatX * odims[2],
                           local[1] * blocksPerMatY * odims[3],
                           1);

            dims_t _odims = {{odims[0], odims[1], odims[2], odims[3]}};
            dims_t _idims = {{idims[0], idims[1], idims[2], idims[3]}};
            dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
            dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};

            tileOp(EnqueueArgs(getQueue(0), global, local),
                   out, in, _odims, _idims, _ostrides, _istrides, offset, blocksPerMatX, blocksPerMatY);

        }
    }
}
