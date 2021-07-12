/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <common/traits.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/copy.hpp>
#include <kernel_headers/memcopy.hpp>
#include <traits.hpp>

#include <algorithm>
#include <string>
#include <vector>

using std::string;
using std::vector;

namespace opencl {
namespace kernel {
typedef struct {
    int dims[4];
} dims_t;

// Push all linear columns to the front, to improve occupancy
// In case of copy operation, input & output parameters needs to be provided!!
// main dims: mdims, mstrides, mndims
// optional dims: odims, ostrides
// ALL parameters will be updated!!
template<bool RESHAPE = false>
void serializeArray(int mdims[4], int mstrides[4], int &mndims, int ostrides[4],
                    int odims[4] = nullptr) noexcept {
    if (RESHAPE) assert(odims != nullptr);
    for (int c = 0; c < mndims - 1; ++c) {
        if (mdims[c] == 1) {
            // Columns with 1 can always be removed.
            // strides of the last column are not updated, because:
            //    - dimension always becomes 1
            //    - strides are therefor not used
            for (int i = c; i < mndims - 1; ++i) {
                mdims[i] = mdims[i + 1];
                if (RESHAPE) odims[i] = odims[i + 1];
                mstrides[i] = mstrides[i + 1];
                ostrides[i] = ostrides[i + 1];
            }
            --mndims;
            mdims[mndims] = 1;
            if (RESHAPE) odims[mndims] = 1;
            --c;  // Redo this column, since it is removed now
        } else if (mdims[c] * mstrides[c] == mstrides[c + 1] &&
                   mdims[c] * ostrides[c] == ostrides[c + 1]) {
            // Combine columns, since they are linear
            // This will increase the dimension of the resulting column,
            // given more opportunities for kernel optimization
            mdims[c] *= mdims[c + 1];
            if (RESHAPE) odims[c] *= odims[c + 1];
            for (int i = c + 1; i < mndims - 1; ++i) {
                mdims[i] = mdims[i + 1];
                if (RESHAPE) odims[i] = odims[i + 1];
                mstrides[i] = mstrides[i + 1];
                ostrides[i] = ostrides[i + 1];
            }
            --mndims;
            mdims[mndims] = 1;
            if (RESHAPE) odims[mndims] = 1;
            --c;  // Redo this colum, since it is removed now
        }
    }
}

// To increase the workload inside a kernel, we move a part of the ndims
// dimension to the last one.  Since this not covered by WG or warp, this is
// always executed as an internal loop.
// In case of copy operation, input & output parameters needs to be provided!!
// main dims: mdims, mstrides, mndims
// optional dims: odims, ostrides
// ALL parameters will be updated!!
void inline increaseWorkload(int elements, int mdims[4], int mstrides[4],
                             int &mndims, int ostrides[4]) noexcept {
    if (elements >= 8192 * 2 && mndims != AF_MAX_DIMS && mndims != 0) {
        // Start only increasing the workload, when all available threads are
        // occupied.

        // list is sorted according to performance improvement
        // 3x looping is faster than 4x, 2x remains faster than no looping
        for (const int i : {3, 4, 5, 7, 11, 2}) {
            if (elements >= 8192 * i && (mdims[mndims - 1] % i) == 0) {
                mdims[mndims - 1] /= i;
                mdims[AF_MAX_DIMS - 1] = i;
                const int mstride = mdims[mndims - 1] * mstrides[mndims - 1];
                const int ostride = mdims[mndims - 1] * ostrides[mndims - 1];
                for (int c = mndims; c < AF_MAX_DIMS; ++c) {
                    mstrides[c] = mstride;
                    ostrides[c] = ostride;
                }
                mndims = AF_MAX_DIMS;
                break;  // Perform this operation only once.
            }
        }
    }
}

template<typename T>
void memcopy(const cl::Buffer &b_out, const dim4 &ostrides,
             const cl::Buffer &b_in, const dim4 &idims, const dim4 &istrides,
             const dim_t ioffset, const dim_t indims, const dim_t ooffset = 0) {
    dims_t idims_{
        static_cast<int>(idims.dims[0]), static_cast<int>(idims.dims[1]),
        static_cast<int>(idims.dims[2]), static_cast<int>(idims.dims[3])};
    dims_t istrides_{
        static_cast<int>(istrides.dims[0]), static_cast<int>(istrides.dims[1]),
        static_cast<int>(istrides.dims[2]), static_cast<int>(istrides.dims[3])};
    dims_t ostrides_{
        static_cast<int>(ostrides.dims[0]), static_cast<int>(ostrides.dims[1]),
        static_cast<int>(ostrides.dims[2]), static_cast<int>(ostrides.dims[3])};
    int indims_ = static_cast<int>(indims);

    bool isLinear = true;
    int elements  = (indims_ == 0) ? 0 : 1;
    for (dim_t dim = 0; dim < indims_; ++dim) {
        isLinear &= (elements == istrides_.dims[dim]) &
                    (elements == ostrides_.dims[dim]);
        elements *= idims_.dims[dim];
    }
    if (elements > 0) {
        if (isLinear) {
            // Both input and output arrays are linear
            getQueue().enqueueCopyBuffer(
                b_in, b_out, ioffset * sizeof(T), ooffset * sizeof(T),
                elements * sizeof(T), nullptr, nullptr);
        } else {
            const vector<TemplateArg> targs = {
                TemplateTypename<T>(),
            };
            const vector<string> options = {
                DefineKeyValue(T, dtype_traits<T>::getName()),
                {getTypeBuildDefinition<T>()},
            };
            auto memCopy =
                common::getKernel("memCopy", {memcopy_cl_src}, targs, options);
            const cl::Device dev = opencl::getDevice();
            const unsigned WG    = static_cast<unsigned>(
                memCopy.get()
                    .getWorkGroupInfo<
                        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(dev));

            serializeArray(idims_.dims, istrides_.dims, indims_,
                           ostrides_.dims);
            increaseWorkload(elements, idims_.dims, istrides_.dims, indims_,
                             ostrides_.dims);

            const cl::NDRange local =
                bestBlockSize<cl::NDRange>(idims_.dims, WG);
            const cl::NDRange global(
                local[0] * divup(idims_.dims[0], local[0]),
                local[1] * divup(idims_.dims[1], local[1]),
                local[2] * divup(idims_.dims[2], local[2]));
            memCopy(cl::EnqueueArgs(getQueue(), global, local), b_out,
                    ostrides_, static_cast<int>(ooffset), b_in, idims_,
                    istrides_, static_cast<int>(ioffset));
            CL_DEBUG_FINISH(getQueue());
        }
    }
}

class BufferPlus {
   public:
    const cl::Buffer *data;
    const dim4 &idims;
    const dim4 &istrides;
    const dim_t ioffset;
    const dim_t ooffset;
    BufferPlus(const cl::Buffer *d, const dim4 &id, const dim4 &is,
               const dim_t io, const dim_t oo)
        : data(d), idims(id), istrides(is), ioffset(io), ooffset(oo){};
};

template<typename T>
void memcopyN(const cl::Buffer &b_out, const dim4 &ostrides,
              const vector<BufferPlus> &ins) {
    Kernel memCopy;
    bool loadKernel = true;
    unsigned WG;
    for (auto &in : ins) {
        dims_t idims_{static_cast<int>(in.idims.dims[0]),
                      static_cast<int>(in.idims.dims[1]),
                      static_cast<int>(in.idims.dims[2]),
                      static_cast<int>(in.idims.dims[3])};
        dims_t istrides_{static_cast<int>(in.istrides.dims[0]),
                         static_cast<int>(in.istrides.dims[1]),
                         static_cast<int>(in.istrides.dims[2]),
                         static_cast<int>(in.istrides.dims[3])};
        dims_t ostrides_{static_cast<int>(ostrides.dims[0]),
                         static_cast<int>(ostrides.dims[1]),
                         static_cast<int>(ostrides.dims[2]),
                         static_cast<int>(ostrides.dims[3])};
        int indims_   = idims_.dims[3] > 1   ? 4
                        : idims_.dims[2] > 1 ? 3
                        : idims_.dims[1] > 1 ? 2
                        : idims_.dims[0] > 0 ? 1
                                             : 0;
        bool isLinear = true;
        int elements  = (indims_ == 0) ? 0 : 1;
        for (int dim = 0; dim < indims_; ++dim) {
            isLinear &= (elements == istrides_.dims[dim]) &
                        (elements == ostrides_.dims[dim]);
            elements *= idims_.dims[dim];
        }
        if (elements > 0) {
            if (isLinear) {
                // Both input and output arrays are linear
                getQueue().enqueueCopyBuffer(
                    *in.data, b_out, in.ioffset * sizeof(T),
                    in.ooffset * sizeof(T), elements * sizeof(T), nullptr,
                    nullptr);
            } else {
                serializeArray(idims_.dims, istrides_.dims, indims_,
                               ostrides_.dims);
                increaseWorkload(elements, idims_.dims, istrides_.dims, indims_,
                                 ostrides_.dims);

                if (loadKernel) {
                    const vector<TemplateArg> targs = {
                        TemplateTypename<T>(),
                    };
                    const vector<string> options = {
                        DefineKeyValue(T, dtype_traits<T>::getName()),
                        {getTypeBuildDefinition<T>()},
                    };
                    memCopy = common::getKernel("memCopy", {memcopy_cl_src},
                                                targs, options);
                    const cl::Device dev = opencl::getDevice();
                    WG                   = static_cast<unsigned>(
                        memCopy.get()
                            .getWorkGroupInfo<
                                CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(
                                dev));
                    loadKernel = false;
                }

                const cl::NDRange local =
                    bestBlockSize<cl::NDRange>(idims_.dims, WG);
                const cl::NDRange global(
                    local[0] * divup(idims_.dims[0], local[0]),
                    local[1] * divup(idims_.dims[1], local[1]),
                    local[2] * divup(idims_.dims[2], local[2]));

                memCopy(cl::EnqueueArgs(getQueue(), global, local), b_out,
                        ostrides_, static_cast<int>(in.ooffset), *in.data,
                        idims_, istrides_, static_cast<int>(in.ioffset));
                CL_DEBUG_FINISH(getQueue());
            }
        }
    }
}

template<typename inType, typename outType>
void copy(const Param out, const Param in, dim_t ondims,
          const outType default_value, const double factor,
          const bool same_dims) {
    dims_t idims_{
        static_cast<int>(in.info.dims[0]), static_cast<int>(in.info.dims[1]),
        static_cast<int>(in.info.dims[2]), static_cast<int>(in.info.dims[3])};
    dims_t istrides_{static_cast<int>(in.info.strides[0]),
                     static_cast<int>(in.info.strides[1]),
                     static_cast<int>(in.info.strides[2]),
                     static_cast<int>(in.info.strides[3])};
    dims_t odims_{
        static_cast<int>(out.info.dims[0]), static_cast<int>(out.info.dims[1]),
        static_cast<int>(out.info.dims[2]), static_cast<int>(out.info.dims[3])};
    dims_t ostrides_{static_cast<int>(out.info.strides[0]),
                     static_cast<int>(out.info.strides[1]),
                     static_cast<int>(out.info.strides[2]),
                     static_cast<int>(out.info.strides[3])};
    int ondims_  = static_cast<int>(ondims);
    int elements = (ondims_ == 0) ? 0 : 1;
    for (int dim = 0; dim < ondims_; ++dim) elements *= odims_.dims[dim];
    if (elements > 0) {
        serializeArray<true>(odims_.dims, ostrides_.dims, ondims_,
                             istrides_.dims, idims_.dims);

        if (std::is_same<inType, double>::value ||
            std::is_same<inType, cdouble>::value) {
            // Only scale in double precision when the input array is also
            // in double or cdouble, otherwise in single precision (float)
            const vector<TemplateArg> targs = {
                TemplateTypename<inType>(), TemplateTypename<outType>(),
                TemplateArg(same_dims),     TemplateTypename<double>(),
                TemplateArg(factor != 1.0),
            };
            const vector<string> options = {
                DefineKeyValue(inType, dtype_traits<inType>::getName()),
                DefineKeyValue(outType, dtype_traits<outType>::getName()),
                string(" -D inType_" + string(dtype_traits<inType>::getName())),
                string(" -D outType_" +
                       string(dtype_traits<outType>::getName())),
                DefineKeyValue(SAME_DIMS, static_cast<int>(same_dims)),
                string(" -D factorType=double"),
                string((factor != 1.0) ? " -D FACTOR" : " -D NOFACTOR"),
                {getTypeBuildDefinition<inType, outType>()},
            };
            auto copy =
                common::getKernel("reshapeCopy", {copy_cl_src}, targs, options);
            const cl::Device dev = opencl::getDevice();
            const unsigned WG    = static_cast<unsigned>(
                copy.get()
                    .getWorkGroupInfo<
                        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(dev));
            const cl::NDRange local =
                bestBlockSize<cl::NDRange>(odims_.dims, WG);
            const cl::NDRange global(
                local[0] * divup(odims_.dims[0], local[0]),
                local[1] * divup(odims_.dims[1], local[1]),
                local[2] * divup(odims_.dims[2], local[2]));
            copy(cl::EnqueueArgs(getQueue(), global, local), *out.data, odims_,
                 ostrides_, static_cast<uint>(out.info.offset), *in.data,
                 idims_, istrides_, static_cast<uint>(in.info.offset),
                 default_value, static_cast<double>(factor));
        } else {
            const vector<TemplateArg> targs = {
                TemplateTypename<inType>(), TemplateTypename<outType>(),
                TemplateArg(same_dims),     TemplateTypename<float>(),
                TemplateArg(factor != 1.0),
            };
            const vector<string> options = {
                DefineKeyValue(inType, dtype_traits<inType>::getName()),
                DefineKeyValue(outType, dtype_traits<outType>::getName()),
                string(" -D inType_" + string(dtype_traits<inType>::getName())),
                string(" -D outType_" +
                       string(dtype_traits<outType>::getName())),
                DefineKeyValue(SAME_DIMS, static_cast<int>(same_dims)),
                string(" -D factorType=float"),
                string((factor != 1.0) ? " -D FACTOR" : " -D NOFACTOR"),
                {getTypeBuildDefinition<inType, outType>()},
            };
            auto copy =
                common::getKernel("reshapeCopy", {copy_cl_src}, targs, options);
            const cl::Device dev = opencl::getDevice();
            const unsigned WG    = static_cast<unsigned>(
                copy.get()
                    .getWorkGroupInfo<
                        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(dev));
            const cl::NDRange local =
                bestBlockSize<cl::NDRange>(odims_.dims, WG);
            const cl::NDRange global(
                local[0] * divup(odims_.dims[0], local[0]),
                local[1] * divup(odims_.dims[1], local[1]),
                local[2] * divup(odims_.dims[2], local[2]));
            copy(cl::EnqueueArgs(getQueue(), global, local), *out.data, odims_,
                 ostrides_, static_cast<int>(out.info.offset), *in.data, idims_,
                 istrides_, static_cast<int>(in.info.offset), default_value,
                 static_cast<float>(factor));
        }
        CL_DEBUG_FINISH(getQueue());
    }
}
}  // namespace kernel
}  // namespace opencl
