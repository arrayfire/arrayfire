#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cuda
{
    namespace kernel
    {
        // Kernel Launch Config Values
        static const unsigned TX = 32;
        static const unsigned TY = 8;

        ///////////////////////////////////////////////////////////////////////////
        // Wrapper functions
        ///////////////////////////////////////////////////////////////////////////
        template<typename T>
        void sort0(Param<T> sx, Param<unsigned> ix, CParam<T> in, const bool dir)
        {
            thrust::device_ptr<T> sx_ptr = thrust::device_pointer_cast(sx.ptr);
            thrust::device_ptr<unsigned> ix_ptr = thrust::device_pointer_cast(ix.ptr);
            thrust::sequence(ix_ptr, ix_ptr + ix.dims[0]);
            if(dir) {
                thrust::sort_by_key(sx_ptr, sx_ptr + sx.dims[0], ix_ptr);
            } else {
                thrust::sort_by_key(sx_ptr, sx_ptr + sx.dims[0], ix_ptr, thrust::greater<T>());
            }
            POST_LAUNCH_CHECK();
        }
    }
}
