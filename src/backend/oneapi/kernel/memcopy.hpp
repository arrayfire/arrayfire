/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <CL/sycl.hpp>
#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
//#include <common/kernel_cache.hpp>
#include <common/traits.hpp>
#include <debug_oneapi.hpp>
#include <traits.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace oneapi {
namespace kernel {

typedef struct {
    dim_t dim[4];
} dims_t;

template<typename T>
class memCopy {
   public:
    memCopy(sycl::accessor<T> out, dims_t ostrides, sycl::accessor<T> in,
            dims_t idims, dims_t istrides, int offset, int groups_0,
            int groups_1, sycl::stream debug)
        : out_(out)
        , in_(in)
        , ostrides_(ostrides)
        , idims_(idims)
        , istrides_(istrides)
        , offset_(offset)
        , groups_0_(groups_0)
        , groups_1_(groups_1)
        , debug_(debug) {}

    void operator()(sycl::nd_item<2> it) const {
        const int lid0 = it.get_local_id(0);
        const int lid1 = it.get_local_id(1);

        sycl::group gg       = it.get_group();
        const int id2        = gg.get_group_id(0) / groups_0_;
        const int id3        = gg.get_group_id(1) / groups_1_;
        const int group_id_0 = gg.get_group_id(0) - groups_0_ * id2;
        const int group_id_1 = gg.get_group_id(1) - groups_1_ * id3;
        const int id0        = group_id_0 * gg.get_local_range(0) + lid0;
        const int id1        = group_id_1 * gg.get_local_range(1) + lid1;

        T *iptr = in_.get_pointer();
        iptr += offset_;
        // FIXME: Do more work per work group

        T *optr = out_.get_pointer();
        optr += id3 * ostrides_.dim[3] + id2 * ostrides_.dim[2] +
                id1 * ostrides_.dim[1];
        iptr += id3 * istrides_.dim[3] + id2 * istrides_.dim[2] +
                id1 * istrides_.dim[1];

        int istride0 = istrides_.dim[0];
        if (id0 < idims_.dim[0] && id1 < idims_.dim[1] && id2 < idims_.dim[2] &&
            id3 < idims_.dim[3]) {
            optr[id0] = iptr[id0 * istride0];
        }
    }

   protected:
    sycl::accessor<T> out_, in_;
    dims_t ostrides_, idims_, istrides_;
    int offset_, groups_0_, groups_1_;
    sycl::stream debug_;
};

constexpr uint DIM0 = 32;
constexpr uint DIM1 = 8;

template<typename T>
void memcopy(sycl::buffer<T> *out, const dim_t *ostrides,
             const sycl::buffer<T> *in, const dim_t *idims,
             const dim_t *istrides, int offset, uint ndims) {
    dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
    dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};
    dims_t _idims    = {{idims[0], idims[1], idims[2], idims[3]}};

    size_t local_size[2] = {DIM0, DIM1};
    if (ndims == 1) {
        local_size[0] *= local_size[1];
        local_size[1] = 1;
    }

    int groups_0 = divup(idims[0], local_size[0]);
    int groups_1 = divup(idims[1], local_size[1]);

    sycl::range<2> local(local_size[0], local_size[1]);
    sycl::range<2> global(groups_0 * idims[2] * local_size[0],
                          groups_1 * idims[3] * local_size[1]);
    sycl::nd_range<2> ndrange(global, local);

    getQueue().submit([=](sycl::handler &h) {
        auto out_acc = out->get_access(h);
        auto in_acc  = const_cast<sycl::buffer<T> *>(in)->get_access(h);

        sycl::stream debug_stream(2048, 128, h);

        h.parallel_for(ndrange,
                       memCopy<T>(out_acc, _ostrides, in_acc, _idims, _istrides,
                                  offset, groups_0, groups_1, debug_stream));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T>
static T scale(T value, double factor) {
    return (T)(double(value) * factor);
}

template<>
cfloat scale<cfloat>(cfloat value, double factor) {
    return cfloat{static_cast<float>(value.real() * factor),
                  static_cast<float>(value.imag() * factor)};
}

template<>
cdouble scale<cdouble>(cdouble value, double factor) {
    return cdouble{value.real() * factor, value.imag() * factor};
}

template<typename inType, typename outType>
outType convertType(inType value) {
    return static_cast<outType>(value);
}

template<>
char convertType<compute_t<common::half>, char>(compute_t<common::half> value) {
    return (char)((short)value);
}

template<>
compute_t<common::half> convertType<char, compute_t<common::half>>(char value) {
    return compute_t<common::half>(value);
}

template<>
unsigned char convertType<compute_t<common::half>, unsigned char>(
    compute_t<common::half> value) {
    return (unsigned char)((short)value);
}

template<>
compute_t<common::half> convertType<unsigned char, compute_t<common::half>>(
    unsigned char value) {
    return compute_t<common::half>(value);
}

template<>
cdouble convertType<cfloat, cdouble>(cfloat value) {
    return cdouble(value.real(), value.imag());
}

template<>
cfloat convertType<cdouble, cfloat>(cdouble value) {
    return cfloat(value.real(), value.imag());
}

#define OTHER_SPECIALIZATIONS(IN_T)                      \
    template<>                                           \
    cfloat convertType<IN_T, cfloat>(IN_T value) {       \
        return cfloat(static_cast<float>(value), 0.0f);  \
    }                                                    \
                                                         \
    template<>                                           \
    cdouble convertType<IN_T, cdouble>(IN_T value) {     \
        return cdouble(static_cast<double>(value), 0.0); \
    }

OTHER_SPECIALIZATIONS(float)
OTHER_SPECIALIZATIONS(double)
OTHER_SPECIALIZATIONS(int)
OTHER_SPECIALIZATIONS(uint)
OTHER_SPECIALIZATIONS(intl)
OTHER_SPECIALIZATIONS(uintl)
OTHER_SPECIALIZATIONS(short)
OTHER_SPECIALIZATIONS(ushort)
OTHER_SPECIALIZATIONS(uchar)
OTHER_SPECIALIZATIONS(char)
OTHER_SPECIALIZATIONS(common::half)

template<typename inType, typename outType, bool SAMEDIMS>
class reshapeCopy {
   public:
    reshapeCopy(sycl::accessor<outType> dst, KParam oInfo,
                sycl::accessor<inType> src, KParam iInfo, outType default_value,
                float factor, dims_t trgt, int blk_x, int blk_y,
                sycl::stream debug)
        : dst_(dst)
        , src_(src)
        , oInfo_(oInfo)
        , iInfo_(iInfo)
        , default_value_(default_value)
        , factor_(factor)
        , trgt_(trgt)
        , blk_x_(blk_x)
        , blk_y_(blk_y)
        , debug_(debug) {}

    void operator()(sycl::nd_item<2> it) const {
        const uint lx = it.get_local_id(0);
        const uint ly = it.get_local_id(1);

        sycl::group gg  = it.get_group();
        uint gz         = gg.get_group_id(0) / blk_x_;
        uint gw         = gg.get_group_id(1) / blk_y_;
        uint blockIdx_x = gg.get_group_id(0) - (blk_x_)*gz;
        uint blockIdx_y = gg.get_group_id(1) - (blk_y_)*gw;
        uint gx         = blockIdx_x * gg.get_local_range(0) + lx;
        uint gy         = blockIdx_y * gg.get_local_range(1) + ly;

        const inType *srcptr = src_.get_pointer();
        outType *dstptr      = dst_.get_pointer();

        const inType *in =
            srcptr + (gw * iInfo_.strides[3] + gz * iInfo_.strides[2] +
                      gy * iInfo_.strides[1] + iInfo_.offset);
        outType *out =
            dstptr + (gw * oInfo_.strides[3] + gz * oInfo_.strides[2] +
                      gy * oInfo_.strides[1] + oInfo_.offset);

        uint istride0 = iInfo_.strides[0];
        uint ostride0 = oInfo_.strides[0];

        if (gy < oInfo_.dims[1] && gz < oInfo_.dims[2] && gw < oInfo_.dims[3]) {
            int loop_offset = gg.get_local_range(0) * blk_x_;
            bool cond =
                gy < trgt_.dim[1] && gz < trgt_.dim[2] && gw < trgt_.dim[3];
            for (int rep = gx; rep < oInfo_.dims[0]; rep += loop_offset) {
                outType temp = default_value_;
                if (SAMEDIMS || (rep < trgt_.dim[0] && cond)) {
                    temp = convertType<inType, outType>(
                        scale<inType>(in[rep * istride0], factor_));
                }
                out[rep * ostride0] = temp;
            }
        }
    }

   protected:
    sycl::accessor<outType> dst_;
    sycl::accessor<inType> src_;
    KParam oInfo_, iInfo_;
    outType default_value_;
    float factor_;
    dims_t trgt_;
    int blk_x_, blk_y_;
    sycl::stream debug_;
};

template<typename inType, typename outType>
void copy(Param<outType> dst, const Param<inType> src, const int ndims,
          const outType default_value, const double factor,
          const bool same_dims) {
    using std::string;

    sycl::range<2> local(DIM0, DIM1);
    size_t local_size[] = {DIM0, DIM1};

    local_size[0] *= local_size[1];
    if (ndims == 1) { local_size[1] = 1; }

    int blk_x = divup(dst.info.dims[0], local_size[0]);
    int blk_y = divup(dst.info.dims[1], local_size[1]);

    sycl::range<2> global(blk_x * dst.info.dims[2] * DIM0,
                          blk_y * dst.info.dims[3] * DIM1);

    sycl::nd_range<2> ndrange(global, local);

    dims_t trgt_dims;
    if (same_dims) {
        trgt_dims = {{dst.info.dims[0], dst.info.dims[1], dst.info.dims[2],
                      dst.info.dims[3]}};
    } else {
        dim_t trgt_l = std::min(dst.info.dims[3], src.info.dims[3]);
        dim_t trgt_k = std::min(dst.info.dims[2], src.info.dims[2]);
        dim_t trgt_j = std::min(dst.info.dims[1], src.info.dims[1]);
        dim_t trgt_i = std::min(dst.info.dims[0], src.info.dims[0]);
        trgt_dims    = {{trgt_i, trgt_j, trgt_k, trgt_l}};
    }

    getQueue().submit([=](sycl::handler &h) {
        auto dst_acc = dst.data->get_access(h);
        auto src_acc =
            const_cast<sycl::buffer<inType> *>(src.data)->get_access(h);

        sycl::stream debug_stream(2048, 128, h);

        if (same_dims) {
            h.parallel_for(ndrange, reshapeCopy<inType, outType, true>(
                                        dst_acc, dst.info, src_acc, src.info,
                                        default_value, (float)factor, trgt_dims,
                                        blk_x, blk_y, debug_stream));
        } else {
            h.parallel_for(ndrange, reshapeCopy<inType, outType, false>(
                                        dst_acc, dst.info, src_acc, src.info,
                                        default_value, (float)factor, trgt_dims,
                                        blk_x, blk_y, debug_stream));
        }
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
