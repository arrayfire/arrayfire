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
        template<typename T, bool DIR>
        void sort0(Param<T> sx, Param<unsigned> ix, CParam<T> in)
        {
            thrust::device_ptr<T>        sx_ptr = thrust::device_pointer_cast(sx.ptr);
            thrust::device_ptr<unsigned> ix_ptr = thrust::device_pointer_cast(ix.ptr);

            for(dim_type w = 0; w < in.dims[3]; w++) {
                for(dim_type z = 0; z < in.dims[2]; z++) {
                    for(dim_type y = 0; y < in.dims[1]; y++) {

                        dim_type sxOffset = w * sx.strides[3] + z * sx.strides[2]
                                          + y * sx.strides[1];
                        dim_type ixOffset = w * ix.strides[3] + z * ix.strides[2]
                                          + y * ix.strides[1];

                        thrust::sequence(ix_ptr + ixOffset, ix_ptr + ixOffset + ix.dims[0]);
                        if(DIR) {
                            thrust::sort_by_key(sx_ptr + sxOffset, sx_ptr + sxOffset + sx.dims[0],
                                                ix_ptr + ixOffset);
                        } else {
                            thrust::sort_by_key(sx_ptr + sxOffset, sx_ptr + sxOffset + sx.dims[0],
                                                ix_ptr + ixOffset, thrust::greater<T>());
                        }
                    }
                }
            }
            POST_LAUNCH_CHECK();
        }
    }
}
