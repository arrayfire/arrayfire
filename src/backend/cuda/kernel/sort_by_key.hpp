#include <math.hpp>
#include <dispatch.hpp>
#include <Param.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
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
        template<typename Tk, typename Tv, bool DIR>
        void sort0_by_key(Param<Tk> okey, Param<Tv> oval,
                         CParam<Tk> ikey, CParam<Tv> ival)
        {
            thrust::device_ptr<Tk>       okey_ptr = thrust::device_pointer_cast(okey.ptr);
            thrust::device_ptr<Tv>       oval_ptr = thrust::device_pointer_cast(oval.ptr);

            for(dim_type w = 0; w < ikey.dims[3]; w++) {
                dim_type okeyW = w * okey.strides[3];
                dim_type ovalW = w * oval.strides[3];
                for(dim_type z = 0; z < ikey.dims[2]; z++) {
                    dim_type okeyWZ = okeyW + z * okey.strides[2];
                    dim_type ovalWZ = ovalW + z * oval.strides[2];
                    for(dim_type y = 0; y < ikey.dims[1]; y++) {

                        dim_type okeyOffset = okeyWZ + y * okey.strides[1];
                        dim_type ovalOffset = ovalWZ + y * oval.strides[1];

                        if(DIR) {
                            thrust::sort_by_key(okey_ptr + okeyOffset, okey_ptr + okeyOffset + okey.dims[0],
                                                oval_ptr + ovalOffset);
                        } else {
                            thrust::sort_by_key(okey_ptr + okeyOffset, okey_ptr + okeyOffset + okey.dims[0],
                                                oval_ptr + ovalOffset, thrust::greater<Tk>());
                        }
                    }
                }
            }
            POST_LAUNCH_CHECK();
        }
    }
}
