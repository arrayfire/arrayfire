#pragma once

#include <af/defines.h>
#include <kernel_headers/tile.hpp>
#include <cl.hpp>
#include <ctx.hpp>
#include <traits.hpp>
#include <helper.hpp>
#include <sstream>
#include <string>

typedef struct
{
    dim_type dims[4];
} dims_t;

#define divup(a, b) ((a+b-1) / b)

using cl::Buffer;
using cl::Program;
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

        template<typename T>
        void tile(Buffer out, const Buffer in,
                  const dim_type *odims, const dim_type *idims,
                  const dim_type *ostrides, const dim_type *istrides, const dim_type offset)
        {
            Program::Sources setSrc;
            setSrc.emplace_back(tile_cl, tile_cl_len);
            Program prog(getCtx(0), setSrc);

            std::ostringstream options;
            options << " -D T="        << dtype_traits<T>::getName()
                    << " -D dim_type=" << dtype_traits<dim_type>::getName();

            prog.build(options.str().c_str());

            auto tileOp = make_kernel<Buffer, const Buffer, const dims_t, const dims_t,
                                      const dims_t, const dims_t, const dim_type,
                                      const unsigned, const unsigned>
                                      (prog, "tile_kernel");

            NDRange local(TX, TY, 1);

            unsigned blocksPerMatX = divup(odims[0], local[0]);
            unsigned blocksPerMatY = divup(odims[1], local[1]);
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
