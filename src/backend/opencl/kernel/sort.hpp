#pragma once
#include <program.hpp>
#include <traits.hpp>
#include <string>
#include <mutex>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

#include <boost/compute.hpp>
namespace compute = boost::compute;

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

        template<typename T, bool DIR>
        void sort0(Param val, const Param in)
        {
            try {
                compute::command_queue c_queue(getQueue()());

                compute::buffer val_buf(val.data());

                for(dim_type w = 0; w < in.info.dims[3]; w++) {
                    dim_type valW = w * val.info.strides[3];
                    for(dim_type z = 0; z < in.info.dims[2]; z++) {
                        dim_type valWZ = valW + z * val.info.strides[2];
                        for(dim_type y = 0; y < in.info.dims[1]; y++) {

                            dim_type valOffset = valWZ + y * val.info.strides[1];

                            if(DIR) {
                                compute::stable_sort(
                                        compute::make_buffer_iterator<T>(val_buf, valOffset),
                                        compute::make_buffer_iterator<T>(val_buf, valOffset + val.info.dims[0]),
                                        compute::less<T>(), c_queue);
                            } else {
                                compute::stable_sort(
                                        compute::make_buffer_iterator<T>(val_buf, valOffset),
                                        compute::make_buffer_iterator<T>(val_buf, valOffset + val.info.dims[0]),
                                        compute::greater<T>(), c_queue);
                            }
                        }
                    }
                }

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }

        template<typename T, bool DIR>
        void sort0_index(Param val, Param idx, const Param in)
        {
            try {
                compute::command_queue c_queue(getQueue()());

                compute::buffer val_buf(val.data());
                compute::buffer idx_buf(idx.data());

                for(dim_type w = 0; w < in.info.dims[3]; w++) {
                    dim_type valW = w * val.info.strides[3];
                    dim_type idxW = w * idx.info.strides[3];
                    for(dim_type z = 0; z < in.info.dims[2]; z++) {
                        dim_type valWZ = valW + z * val.info.strides[2];
                        dim_type idxWZ = idxW + z * idx.info.strides[2];
                        for(dim_type y = 0; y < in.info.dims[1]; y++) {

                            dim_type valOffset = valWZ + y * val.info.strides[1];
                            dim_type idxOffset = idxWZ + y * idx.info.strides[1];

                            compute::buffer_iterator<unsigned> idx_begin(idx_buf, idxOffset);
                            compute::iota(idx_begin, idx_begin + in.info.dims[0], 0, c_queue);

                            if(DIR) {
                                compute::sort_by_key(
                                        compute::make_buffer_iterator<T>(val_buf, valOffset),
                                        compute::make_buffer_iterator<T>(val_buf, valOffset + val.info.dims[0]),
                                        idx_begin, compute::less<T>(), c_queue);
                            } else {
                                compute::sort_by_key(
                                        compute::make_buffer_iterator<T>(val_buf, valOffset),
                                        compute::make_buffer_iterator<T>(val_buf, valOffset + val.info.dims[0]),
                                        idx_begin, compute::greater<T>(), c_queue);
                            }
                        }
                    }
                }

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }

        template<typename Tk, typename Tv, bool DIR>
        void sort0_by_key(Param okey, Param oval, const Param ikey, const Param ival)
        {
            try {
                compute::command_queue c_queue(getQueue()());

                compute::buffer okey_buf(okey.data());
                compute::buffer oval_buf(oval.data());

                for(dim_type w = 0; w < ikey.info.dims[3]; w++) {
                    dim_type okeyW = w * okey.info.strides[3];
                    dim_type ovalW = w * oval.info.strides[3];
                    for(dim_type z = 0; z < ikey.info.dims[2]; z++) {
                        dim_type okeyWZ = okeyW + z * okey.info.strides[2];
                        dim_type ovalWZ = ovalW + z * oval.info.strides[2];
                        for(dim_type y = 0; y < ikey.info.dims[1]; y++) {

                            dim_type okeyOffset = okeyWZ + y * okey.info.strides[1];
                            dim_type ovalOffset = ovalWZ + y * oval.info.strides[1];

                            if(DIR) {
                                compute::sort_by_key(
                                        compute::make_buffer_iterator<Tk>(okey_buf, okeyOffset),
                                        compute::make_buffer_iterator<Tk>(okey_buf, okeyOffset + okey.info.dims[0]),
                                        compute::make_buffer_iterator<Tv>(oval_buf, ovalOffset),
                                        compute::less<Tk>(), c_queue);
                            } else {
                                compute::sort_by_key(
                                        compute::make_buffer_iterator<Tk>(okey_buf, okeyOffset),
                                        compute::make_buffer_iterator<Tk>(okey_buf, okeyOffset + okey.info.dims[0]),
                                        compute::make_buffer_iterator<Tv>(oval_buf, ovalOffset),
                                        compute::greater<Tk>(), c_queue);
                            }
                        }
                    }
                }

                CL_DEBUG_FINISH(getQueue());
            } catch (cl::Error err) {
                CL_TO_AF_ERROR(err);
                throw;
            }
        }
    }
}
