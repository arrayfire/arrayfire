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
#include <common/kernel_cache.hpp>
#include <common/traits.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/copy.hpp>
#include <kernel_headers/memcopy.hpp>
#include <threadsMgt.hpp>
#include <traits.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {
typedef struct {
    int dims[4];
} dims_type;

// Increase vectorization by increasing the used type up to maxVectorWidth.
// Example:
//  input array<int> with return value = 4, means that the array became
//  array<int4>.
//
// Parameters
//  - IN     maxVectorWidth: maximum vectorisation desired
//  - IN/OUT dims[4]: dimensions of the array
//  - IN/OUT istrides[4]: strides of the input array
//  - IN/OUT indims: ndims of the input array.  Updates when dim[0] becomes 1
//  - IN/OUT ioffset: offset of the input array
//  - IN/OUT ostrides[4]: strides of the output array
//  - IN/OUT ooffset: offset of the output array
//
// Returns
//  - maximum obtained vectorization.
//  - All the parameters are updated accordingly
//
static inline unsigned vectorizeShape(const unsigned maxVectorWidth,
                                      int dims[4], int istrides[4], int& indims,
                                      dim_t& ioffset, int ostrides[4],
                                      dim_t& ooffset) {
    unsigned vectorWidth{1};
    if ((maxVectorWidth != 1) & (istrides[0] == 1) & (ostrides[0] == 1)) {
        // - Only adjacent items can be vectorized into a base vector type
        // - global is the OR of the values to be checked.  When global is
        // divisable by 2, than all source values are also
        // - The buffers are always aligned at 128 Bytes, so the alignment is
        // only dependable on the offsets
        dim_t global{dims[0] | ioffset | ooffset};
        for (int i{1}; i < indims; ++i) { global |= istrides[i] | ostrides[i]; }

        // Determine the maximum vectorization possible
        unsigned count{0};
        while (((global & 1) == 0) & (vectorWidth < maxVectorWidth)) {
            ++count;
            vectorWidth <<= 1;
            global >>= 1;
        }
        if (count != 0) {
            // update the dimensions, to correspond with the new vectorization
            dims[0] >>= count;
            ioffset >>= count;
            ooffset >>= count;
            for (int i{1}; i < indims; ++i) {
                istrides[i] >>= count;
                ostrides[i] >>= count;
            }
            if (dims[0] == 1) {
                // Vectorization has absorbed the full dim0, so eliminate
                // the 1st dimension
                --indims;
                for (int i{0}; i < indims; ++i) {
                    dims[i]     = dims[i + 1];
                    istrides[i] = istrides[i + 1];
                    ostrides[i] = ostrides[i + 1];
                }
                dims[indims] = 1;
            }
        }
    }
    return vectorWidth;
}

template<typename T>
void memcopy(const cl::Buffer& b_out, const dim4& ostrides,
             const cl::Buffer& b_in, const dim4& idims, const dim4& istrides,
             dim_t ioffset, const dim_t indims, dim_t ooffset = 0) {
    dims_type idims_{
        static_cast<int>(idims.dims[0]), static_cast<int>(idims.dims[1]),
        static_cast<int>(idims.dims[2]), static_cast<int>(idims.dims[3])};
    dims_type istrides_{
        static_cast<int>(istrides.dims[0]), static_cast<int>(istrides.dims[1]),
        static_cast<int>(istrides.dims[2]), static_cast<int>(istrides.dims[3])};
    dims_type ostrides_{
        static_cast<int>(ostrides.dims[0]), static_cast<int>(ostrides.dims[1]),
        static_cast<int>(ostrides.dims[2]), static_cast<int>(ostrides.dims[3])};
    int indims_{static_cast<int>(indims)};

    const size_t totalSize{idims.elements() * sizeof(T) * 2};
    removeEmptyColumns(idims_.dims, indims_, ostrides_.dims);
    indims_ =
        removeEmptyColumns(idims_.dims, indims_, idims_.dims, istrides_.dims);
    indims_ =
        combineColumns(idims_.dims, istrides_.dims, indims_, ostrides_.dims);

    // Optimization memory access and caching.
    // Best performance is achieved with the highest vectorization
    // (<int> --> <int2>,<int4>, ...), since more data is processed per IO.
    const cl::Device dev{opencl::getDevice()};
    const unsigned DevicePreferredVectorWidthChar{
        dev.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>()};
    // When the architecture prefers some width's, it is certainly
    // on char.  No preference means vector width 1 returned.
    const bool DevicePreferredVectorWidth{DevicePreferredVectorWidthChar != 1};
    size_t maxVectorWidth{
        DevicePreferredVectorWidth
            ? sizeof(T) == 1 ? DevicePreferredVectorWidthChar
              : sizeof(T) == 2
                  ? dev.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>()
              : sizeof(T) == 4
                  ? dev.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>()
              : sizeof(T) == 8
                  ? dev.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>()
                  : 1
        : sizeof(T) > 8 ? 1
                        : 16 / sizeof(T)};
    const size_t vectorWidth{vectorizeShape(maxVectorWidth, idims_.dims,
                                            istrides_.dims, indims_, ioffset,
                                            ostrides_.dims, ooffset)};
    const size_t sizeofNewT{sizeof(T) * vectorWidth};

    threadsMgt<int> th(idims_.dims, indims_, 1, 1, totalSize, sizeofNewT);
    const char* kernelName{
        th.loop0   ? "memCopyLoop0"
        : th.loop1 ? th.loop3 ? "memCopyLoop13" : "memCopyLoop1"
        : th.loop3 ? "memCopyLoop3"
                   : "memCopy"};  // Conversion to  base vector types.
    const char* tArg{
        sizeofNewT == 1   ? "char"
        : sizeofNewT == 2 ? "short"
        : sizeofNewT == 4 ? "float"
        : sizeofNewT == 8 ? "float2"
        : sizeofNewT == 16
            ? "float4"
            : "type is larger than 16 bytes, which is unsupported"};
    auto memCopy =
        common::getKernel(kernelName, {{memcopy_cl_src}}, TemplateArgs(tArg),
                          {{DefineKeyValue(T, tArg)}});
    const cl::NDRange local{th.genLocal(memCopy.get())};
    const cl::NDRange global{th.genGlobal(local)};

    memCopy(cl::EnqueueArgs(getQueue(), global, local), b_out, ostrides_,
            static_cast<int>(ooffset), b_in, idims_, istrides_,
            static_cast<int>(ioffset));
    CL_DEBUG_FINISH(getQueue());
}

template<typename inType, typename outType>
void copy(const Param out, const Param in, dim_t ondims,
          const outType default_value, const double factor) {
    dims_type idims_{
        static_cast<int>(in.info.dims[0]), static_cast<int>(in.info.dims[1]),
        static_cast<int>(in.info.dims[2]), static_cast<int>(in.info.dims[3])};
    dims_type istrides_{static_cast<int>(in.info.strides[0]),
                        static_cast<int>(in.info.strides[1]),
                        static_cast<int>(in.info.strides[2]),
                        static_cast<int>(in.info.strides[3])};
    dims_type odims_{
        static_cast<int>(out.info.dims[0]), static_cast<int>(out.info.dims[1]),
        static_cast<int>(out.info.dims[2]), static_cast<int>(out.info.dims[3])};
    dims_type ostrides_{static_cast<int>(out.info.strides[0]),
                        static_cast<int>(out.info.strides[1]),
                        static_cast<int>(out.info.strides[2]),
                        static_cast<int>(out.info.strides[3])};
    int ondims_{static_cast<int>(ondims)};
    const size_t totalSize{odims_.dims[0] * odims_.dims[1] * odims_.dims[2] *
                               odims_.dims[3] * sizeof(outType) +
                           idims_.dims[0] * idims_.dims[1] * idims_.dims[2] *
                               idims_.dims[3] * sizeof(inType)};
    bool same_dims{true};
    for (int i{0}; i < ondims_; ++i) {
        if (idims_.dims[i] > odims_.dims[i]) {
            idims_.dims[i] = odims_.dims[i];
        } else if (idims_.dims[i] != odims_.dims[i]) {
            same_dims = false;
        }
    }

    removeEmptyColumns(odims_.dims, ondims_, idims_.dims, istrides_.dims);
    ondims_ =
        removeEmptyColumns(odims_.dims, ondims_, odims_.dims, ostrides_.dims);
    ondims_ = combineColumns(odims_.dims, ostrides_.dims, ondims_, idims_.dims,
                             istrides_.dims);

    constexpr int factorTypeIdx{std::is_same<inType, double>::value ||
                                std::is_same<inType, cdouble>::value};
    const char* factorType[]{"float", "double"};

    const std::vector<TemplateArg> targs{
        TemplateTypename<inType>(), TemplateTypename<outType>(),
        TemplateArg(same_dims),     TemplateArg(factorType[factorTypeIdx]),
        TemplateArg(factor != 1.0),
    };
    const std::vector<std::string> options{
        DefineKeyValue(inType, dtype_traits<inType>::getName()),
        DefineKeyValue(outType, dtype_traits<outType>::getName()),
        std::string(" -D inType_") + dtype_traits<inType>::getName(),
        std::string(" -D outType_") + dtype_traits<outType>::getName(),
        DefineKeyValue(SAME_DIMS, static_cast<int>(same_dims)),
        std::string(" -D factorType=") + factorType[factorTypeIdx],
        std::string((factor != 1.0) ? " -D FACTOR" : " -D NOFACTOR"),
        {getTypeBuildDefinition<inType, outType>()},
    };

    threadsMgt<int> th(odims_.dims, ondims_, 1, 1, totalSize, sizeof(outType));
    auto copy = common::getKernel(th.loop0   ? "scaledCopyLoop0"
                                  : th.loop3 ? "scaledCopyLoop13"
                                  : th.loop1 ? "scaledCopyLoop1"
                                             : "scaledCopy",
                                  {{copy_cl_src}}, targs, options);
    const cl::NDRange local{th.genLocal(copy.get())};
    const cl::NDRange global{th.genGlobal(local)};

    if (factorTypeIdx == 0) {
        copy(cl::EnqueueArgs(getQueue(), global, local), *out.data, odims_,
             ostrides_, static_cast<uint>(out.info.offset), *in.data, idims_,
             istrides_, static_cast<uint>(in.info.offset), default_value,
             static_cast<float>(factor));
    } else {
        copy(cl::EnqueueArgs(getQueue(), global, local), *out.data, odims_,
             ostrides_, static_cast<uint>(out.info.offset), *in.data, idims_,
             istrides_, static_cast<uint>(in.info.offset), default_value,
             static_cast<double>(factor));
    }

    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
