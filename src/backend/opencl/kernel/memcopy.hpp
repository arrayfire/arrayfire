/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <kernel_headers/memcopy.hpp>
#include <kernel_headers/copy.hpp>
#include <program.hpp>
#include <traits.hpp>
#include <sstream>
#include <string>
#include <map>
#include <algorithm>
#include <dispatch.hpp>
#include <Param.hpp>
#include <debug_opencl.hpp>

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
        dim_t dim[4];
    } dims_t;

    static const uint DIM0 = 32;
    static const uint DIM1 =  8;

    template<typename T>
    void memcopy(cl::Buffer out, const dim_t *ostrides,
                 const cl::Buffer in, const dim_t *idims,
                 const dim_t *istrides, int offset, uint ndims)
    {
        try {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static std::map<int, Program*>    cpyProgs;
            static std::map<int, Kernel*>   cpyKernels;

            int device = getActiveDeviceId();

            std::call_once(compileFlags[device], [&]() {
                std::ostringstream options;
                options << " -D T=" << dtype_traits<T>::getName();
                if (std::is_same<T, double>::value ||
                    std::is_same<T, cdouble>::value) {
                    options << " -D USE_DOUBLE";
                }
                Program prog;
                buildProgram(prog, memcopy_cl, memcopy_cl_len, options.str());
                cpyProgs[device]   = new Program(prog);
                cpyKernels[device] = new Kernel(*cpyProgs[device], "memcopy_kernel");
            });

            dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
            dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};
            dims_t _idims = {{idims[0], idims[1], idims[2], idims[3]}};

            size_t local_size[2] = {DIM0, DIM1};
            if (ndims == 1) {
                local_size[0] *= local_size[1];
                local_size[1]  = 1;
            }

            int groups_0 = divup(idims[0], local_size[0]);
            int groups_1 = divup(idims[1], local_size[1]);

            NDRange local(local_size[0], local_size[1]);
            NDRange global(groups_0 * idims[2] * local_size[0],
                           groups_1 * idims[3] * local_size[1]);

            auto memcopy_kernel = make_kernel< Buffer, dims_t,
                                               Buffer, dims_t,
                                               dims_t, int,
                                               int, int >(*cpyKernels[device]);

            memcopy_kernel(EnqueueArgs(getQueue(), global, local),
                out, _ostrides, in, _idims, _istrides, offset, groups_0, groups_1);
            CL_DEBUG_FINISH(getQueue());
        }
        catch (cl::Error err) {
            CL_TO_AF_ERROR(err);
            throw;
        }
    }

    template<typename inType, typename outType, bool same_dims>
    void copy(Param dst, const Param src, int ndims, outType default_value, double factor)
    {
        try {
            static std::once_flag compileFlags[DeviceManager::MAX_DEVICES];
            static std::map<int, Program*>    cpyProgs;
            static std::map<int, Kernel*>   cpyKernels;

            int device = getActiveDeviceId();

            std::call_once(compileFlags[device], [&]() {

                        std::ostringstream options;
                        options << " -D inType=" << dtype_traits<inType>::getName()
                            << " -D outType=" << dtype_traits<outType>::getName()
                            << " -D inType_" << dtype_traits<inType>::getName()
                            << " -D outType_" << dtype_traits<outType>::getName()
                            << " -D SAME_DIMS=" << same_dims;
                        if (std::is_same<inType, double>::value  ||
                            std::is_same<inType, cdouble>::value ||
                            std::is_same<outType, double>::value ||
                            std::is_same<outType, cdouble>::value) {
                            options << " -D USE_DOUBLE";
                        }

                        Program prog;
                        buildProgram(prog, copy_cl, copy_cl_len, options.str());
                        cpyProgs[device]   = new Program(prog);
                        cpyKernels[device] = new Kernel(*cpyProgs[device], "copy");
                    });

            NDRange local(DIM0, DIM1);
            size_t local_size[] = {DIM0, DIM1};

            local_size[0] *= local_size[1];
            if (ndims == 1) {
                local_size[1] = 1;
            }

            int blk_x = divup(dst.info.dims[0], local_size[0]);
            int blk_y = divup(dst.info.dims[1], local_size[1]);

            NDRange global(blk_x * dst.info.dims[2] * DIM0,
                    blk_y * dst.info.dims[3] * DIM1);

            dims_t trgt_dims;
            if (same_dims) {
                trgt_dims= {{dst.info.dims[0], dst.info.dims[1], dst.info.dims[2], dst.info.dims[3]}};
            } else {
                dim_t trgt_l = std::min(dst.info.dims[3], src.info.dims[3]);
                dim_t trgt_k = std::min(dst.info.dims[2], src.info.dims[2]);
                dim_t trgt_j = std::min(dst.info.dims[1], src.info.dims[1]);
                dim_t trgt_i = std::min(dst.info.dims[0], src.info.dims[0]);
                trgt_dims= {{trgt_i, trgt_j, trgt_k, trgt_l}};
            }

            auto copyOp = make_kernel<Buffer, KParam, Buffer, KParam,
                                      outType, float, dims_t,
                                      int, int
                                     >(*cpyKernels[device]);

            copyOp(EnqueueArgs(getQueue(), global, local),
                   *dst.data, dst.info, *src.data, src.info,
                   default_value, (float)factor, trgt_dims, blk_x, blk_y);
            CL_DEBUG_FINISH(getQueue());
        } catch (cl::Error err) {
            CL_TO_AF_ERROR(err);
            throw;
        }
    }

}

}
