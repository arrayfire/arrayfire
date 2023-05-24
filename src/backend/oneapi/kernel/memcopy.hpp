/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <common/traits.hpp>
#include <debug_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <sycl/sycl.hpp>
#include <traits.hpp>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
using factortypes = typename std::conditional<std::is_same_v<T, double> ||
                                                  std::is_same_v<T, cdouble>,
                                              double, float>::type;

template<typename T, typename FACTORTYPE = factortypes<T>>
inline T scale(T value, FACTORTYPE factor) {
    return (T)(FACTORTYPE(value) * factor);
}

template<>
inline cfloat scale<cfloat, float>(cfloat value, float factor) {
    return cfloat{static_cast<float>(value.real() * factor),
                  static_cast<float>(value.imag() * factor)};
}

template<>
inline cdouble scale<cdouble, double>(cdouble value, double factor) {
    return cdouble{value.real() * factor, value.imag() * factor};
}

typedef struct {
    dim_t dim[4];
} dims_t;

template<typename T>
class memCopy {
   public:
    memCopy(write_accessor<T> out, dims_t ostrides, int ooffset,
            read_accessor<T> in, dims_t idims, dims_t istrides, int ioffset,
            int groups_0, int groups_1)
        : out_(out)
        , ostrides_(ostrides)
        , ooffset_(ooffset)
        , in_(in)
        , idims_(idims)
        , istrides_(istrides)
        , ioffset_(ioffset)
        , groups_0_(groups_0)
        , groups_1_(groups_1) {}

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

        const T *iptr = in_.get_pointer();
        // FIXME: Do more work per work group

        T *optr = out_.get_pointer();
        optr += id3 * ostrides_.dim[3] + id2 * ostrides_.dim[2] +
                id1 * ostrides_.dim[1] + ooffset_;
        iptr += id3 * istrides_.dim[3] + id2 * istrides_.dim[2] +
                id1 * istrides_.dim[1] + ioffset_;

        int istride0 = istrides_.dim[0];
        size_t idd0  = idims_.dim[0];
        size_t idd1  = idims_.dim[1];
        size_t idd2  = idims_.dim[2];
        size_t idd3  = idims_.dim[3];

        if (id0 < idd0 && id1 < idd1 && id2 < idd2 && id3 < idd3) {
            optr[id0] = iptr[id0 * istride0];
        }
    }

   protected:
    write_accessor<T> out_;
    dims_t ostrides_;
    int ooffset_;
    read_accessor<T> in_;
    dims_t idims_, istrides_;
    int ioffset_, groups_0_, groups_1_;
};

constexpr uint DIM0 = 32;
constexpr uint DIM1 = 8;

template<typename T>
void memcopy(sycl::buffer<T> *out, const dim_t *ostrides,
             const sycl::buffer<T> *in, const dim_t *idims,
             const dim_t *istrides, dim_t ioffset, uint indims,
             dim_t ooffset = 0) {
    dims_t _ostrides = {{ostrides[0], ostrides[1], ostrides[2], ostrides[3]}};
    dims_t _istrides = {{istrides[0], istrides[1], istrides[2], istrides[3]}};
    dims_t _idims    = {{idims[0], idims[1], idims[2], idims[3]}};

    size_t local_size[2] = {DIM0, DIM1};
    if (indims == 1) {
        local_size[0] *= local_size[1];
        local_size[1] = 1;
    }

    int groups_0 = divup(idims[0], local_size[0]);
    int groups_1 = divup(idims[1], local_size[1]);

    sycl::range<2> local(local_size[0], local_size[1]);
    sycl::range<2> global(groups_0 * idims[2] * local_size[0],
                          groups_1 * idims[3] * local_size[1]);
    sycl::nd_range<2> ndrange(global, local);

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<T> out_acc{*out, h};
        read_accessor<T> in_acc{*const_cast<sycl::buffer<T> *>(in), h};

        h.parallel_for(ndrange,
                       memCopy<T>(out_acc, _ostrides, ooffset, in_acc, _idims,
                                  _istrides, ioffset, groups_0, groups_1));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename inType, typename outType>
inline outType convertType(inType value) {
    return static_cast<outType>(value);
}

template<>
inline char convertType<compute_t<arrayfire::common::half>, char>(
    compute_t<arrayfire::common::half> value) {
    return (char)((short)value);
}

template<>
inline compute_t<arrayfire::common::half>
convertType<char, compute_t<arrayfire::common::half>>(char value) {
    return compute_t<arrayfire::common::half>(value);
}

template<>
unsigned char inline convertType<compute_t<arrayfire::common::half>,
                                 unsigned char>(
    compute_t<arrayfire::common::half> value) {
    return (unsigned char)((short)value);
}

template<>
inline compute_t<arrayfire::common::half>
convertType<unsigned char, compute_t<arrayfire::common::half>>(
    unsigned char value) {
    return compute_t<arrayfire::common::half>(value);
}

#define OTHER_SPECIALIZATIONS(IN_T)                         \
    template<>                                              \
    inline cfloat convertType<IN_T, cfloat>(IN_T value) {   \
        return cfloat(static_cast<float>(value), 0.0f);     \
    }                                                       \
                                                            \
    template<>                                              \
    inline cdouble convertType<IN_T, cdouble>(IN_T value) { \
        return cdouble(static_cast<double>(value), 0.0);    \
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
OTHER_SPECIALIZATIONS(arrayfire::common::half)

template<typename inType, typename outType, bool SAMEDIMS>
class reshapeCopy {
   public:
    reshapeCopy(write_accessor<outType> dst, KParam oInfo,
                read_accessor<inType> src, KParam iInfo, outType default_value,
                factortypes<inType> factor, dims_t trgt, int blk_x, int blk_y)
        : dst_(dst)
        , src_(src)
        , oInfo_(oInfo)
        , iInfo_(iInfo)
        , default_value_(default_value)
        , factor_(factor)
        , trgt_(trgt)
        , blk_x_(blk_x)
        , blk_y_(blk_y) {}

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

        size_t odims0 = oInfo_.dims[0];
        size_t odims1 = oInfo_.dims[1];
        size_t odims2 = oInfo_.dims[2];
        size_t odims3 = oInfo_.dims[3];

        size_t tdims0 = trgt_.dim[0];
        size_t tdims1 = trgt_.dim[1];
        size_t tdims2 = trgt_.dim[2];
        size_t tdims3 = trgt_.dim[3];

        if (gy < odims1 && gz < odims2 && gw < odims3) {
            int loop_offset = gg.get_local_range(0) * blk_x_;
            bool cond       = gy < tdims1 && gz < tdims2 && gw < tdims3;
            for (int rep = gx; rep < odims0; rep += loop_offset) {
                outType temp = default_value_;
                if (SAMEDIMS || (rep < tdims0 && cond)) {
                    temp = convertType<inType, outType>(
                        scale<inType>(in[rep * istride0], factor_));
                }
                out[rep * ostride0] = temp;
            }
        }
    }

   protected:
    write_accessor<outType> dst_;
    read_accessor<inType> src_;
    KParam oInfo_, iInfo_;
    outType default_value_;
    factortypes<inType> factor_;
    dims_t trgt_;
    int blk_x_, blk_y_;
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

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<outType> dst_acc{*dst.data, h};
        read_accessor<inType> src_acc{
            *const_cast<sycl::buffer<inType> *>(src.data), h};

        if (same_dims) {
            h.parallel_for(ndrange,
                           reshapeCopy<inType, outType, true>(
                               dst_acc, dst.info, src_acc, src.info,
                               default_value, factor, trgt_dims, blk_x, blk_y));
        } else {
            h.parallel_for(ndrange,
                           reshapeCopy<inType, outType, false>(
                               dst_acc, dst.info, src_acc, src.info,
                               default_value, factor, trgt_dims, blk_x, blk_y));
        }
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
