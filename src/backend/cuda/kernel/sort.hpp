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
        void sort0(Param<T> val, CParam<T> in)
        {
            thrust::device_ptr<T> val_ptr = thrust::device_pointer_cast(val.ptr);

            for(dim_type w = 0; w < in.dims[3]; w++) {
                dim_type valW = w * val.strides[3];
                for(dim_type z = 0; z < in.dims[2]; z++) {
                    dim_type valWZ = valW + z * val.strides[2];
                    for(dim_type y = 0; y < in.dims[1]; y++) {

                        dim_type valOffset = valWZ + y * val.strides[1];

                        if(DIR) {
                            thrust::stable_sort(val_ptr + valOffset, val_ptr + valOffset + val.dims[0]);
                        } else {
                            thrust::stable_sort(val_ptr + valOffset, val_ptr + valOffset + val.dims[0],
                                                thrust::greater<T>());
                        }
                    }
                }
            }
            POST_LAUNCH_CHECK();
        }

        template<typename T, bool DIR>
        void sort0_index(Param<T> val, Param<unsigned> idx, CParam<T> in)
        {
            thrust::device_ptr<T>        val_ptr = thrust::device_pointer_cast(val.ptr);
            thrust::device_ptr<unsigned> idx_ptr = thrust::device_pointer_cast(idx.ptr);

            for(dim_type w = 0; w < in.dims[3]; w++) {
                dim_type valW = w * val.strides[3];
                dim_type idxW = w * idx.strides[3];
                for(dim_type z = 0; z < in.dims[2]; z++) {
                    dim_type valWZ = valW + z * val.strides[2];
                    dim_type idxWZ = idxW + z * idx.strides[2];
                    for(dim_type y = 0; y < in.dims[1]; y++) {

                        dim_type valOffset = valWZ + y * val.strides[1];
                        dim_type idxOffset = idxWZ + y * idx.strides[1];

                        thrust::sequence(idx_ptr + idxOffset, idx_ptr + idxOffset + idx.dims[0]);
                        if(DIR) {
                            thrust::sort_by_key(val_ptr + valOffset, val_ptr + valOffset + val.dims[0],
                                                idx_ptr + idxOffset);
                        } else {
                            thrust::sort_by_key(val_ptr + valOffset, val_ptr + valOffset + val.dims[0],
                                                idx_ptr + idxOffset, thrust::greater<T>());
                        }
                    }
                }
            }
            POST_LAUNCH_CHECK();
        }
    }
}
