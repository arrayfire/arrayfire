#include <kernel_headers/memcopy.hpp>
#define __CL_ENABLE_EXCEPTIONS
#include <cl.hpp>
#include <ctx.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>
#include <dispatch.hpp>

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
    typedef struct
    {
        dim_type dim[4];
    } dims_t;

    static const uint DIM0 = 32;
    static const uint DIM1 =  8;

    template<typename T>
    void memcopy(cl::Buffer out, const dim_type *ostrides,
                 const cl::Buffer in, const dim_type *idims,
                 const dim_type *istrides, dim_type offset, uint ndims)
    {

        dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
        dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};
        dims_t _idims = {{idims[0], idims[1], idims[2], idims[3]}};

        size_t local_size[2] = {DIM0, DIM1};
        if (ndims == 1) {
            local_size[0] *= local_size[1];
            local_size[1]  = 1;
       }

        dim_type groups_0 = divup(idims[0], local_size[0]);
        dim_type groups_1 = divup(idims[1], local_size[1]);

        NDRange local(local_size[0], local_size[1]);
        NDRange global(groups_0 * idims[2] * local_size[0],
                       groups_1 * idims[3] * local_size[1]);

        Program::Sources src;
        src.emplace_back(memcopy_cl, memcopy_cl_len);

        Program prog(getCtx(0), src);

        std::ostringstream options;
        options << " -D T=" << dtype_traits<T>::getName()
                << " -D dim_type=" << dtype_traits<dim_type>::getName();

        prog.build(options.str().c_str());

        auto memcopy_kernel = make_kernel< Buffer, dims_t,
                                           Buffer, dims_t,
                                           dims_t, dim_type,
                                           uint, uint >(prog, "memcopy_kernel");

        memcopy_kernel(EnqueueArgs(getQueue(0), global, local),
                       out, _ostrides, in, _idims, _istrides, offset, groups_0, groups_1);
    }
}
}
