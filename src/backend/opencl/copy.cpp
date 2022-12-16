/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <copy.hpp>
#include <kernel/memcopy.hpp>

#include <Array.hpp>
#include <common/complex.hpp>
#include <common/half.hpp>
#include <err_opencl.hpp>
#include <math.hpp>

using arrayfire::common::half;
using arrayfire::common::is_complex;

namespace arrayfire {
namespace opencl {

template<typename T>
void copyData(T *data, const Array<T> &src) {
    if (src.elements() > 0) {
        Array<T> out = src.isReady() && src.isLinear() ? src : copyArray(src);
        // out is now guaranteed linear
        getQueue().enqueueReadBuffer(*out.get(), CL_TRUE,
                                     sizeof(T) * out.getOffset(),
                                     sizeof(T) * out.elements(), data);
    }
}

template<typename T>
Array<T> copyArray(const Array<T> &src) {
    Array<T> out = createEmptyArray<T>(src.dims());
    if (src.elements() > 0) {
        if (src.isReady()) {
            if (src.isLinear()) {
                getQueue().enqueueCopyBuffer(
                    *src.get(), *out.get(), src.getOffset() * sizeof(T), 0,
                    src.elements() * sizeof(T), nullptr, nullptr);
            } else {
                kernel::memcopy<T>(*out.get(), out.strides(), *src.get(),
                                   src.dims(), src.strides(), src.getOffset(),
                                   src.ndims());
            }
        } else {
            Param info = {out.get(),
                          {{src.dims().dims[0], src.dims().dims[1],
                            src.dims().dims[2], src.dims().dims[3]},
                           {out.strides().dims[0], out.strides().dims[1],
                            out.strides().dims[2], out.strides().dims[3]},
                           0}};
            evalNodes(info, src.getNode().get());
        }
    }
    return out;
}

template<typename T>
void multiply_inplace(Array<T> &src, double norm) {
    if (src.elements() > 0) {
        kernel::copy<T, T>(src, src, src.ndims(), scalar<T>(0), norm);
    }
}

template<typename inType, typename outType>
struct copyWrapper {
    void operator()(Array<outType> &dst, Array<inType> const &src) {
        kernel::copy<inType, outType>(dst, src, dst.ndims(), scalar<outType>(0),
                                      1.0);
    }
};

template<typename T>
struct copyWrapper<T, T> {
    void operator()(Array<T> &dst, Array<T> const &src) {
        if (src.elements() > 0) {
            if (dst.dims() == src.dims()) {
                if (src.isReady()) {
                    if (dst.isLinear() && src.isLinear()) {
                        getQueue().enqueueCopyBuffer(
                            *src.get(), *dst.get(), src.getOffset() * sizeof(T),
                            dst.getOffset() * sizeof(T),
                            src.elements() * sizeof(T), nullptr, nullptr);
                    } else {
                        kernel::memcopy<T>(*dst.get(), dst.strides(),
                                           *src.get(), src.dims(),
                                           src.strides(), src.getOffset(),
                                           src.ndims(), dst.getOffset());
                    }
                } else {
                    Param info = {
                        dst.get(),
                        {{src.dims().dims[0], src.dims().dims[1],
                          src.dims().dims[2], src.dims().dims[3]},
                         {dst.strides().dims[0], dst.strides().dims[1],
                          dst.strides().dims[2], dst.strides().dims[3]},
                         dst.getOffset()}};
                    evalNodes(info, src.getNode().get());
                }
            } else {
                // dst has more elements than src, so default has to be applied
                kernel::copy<T, T>(dst, src, dst.ndims(), scalar<T>(0), 1.0);
            }
        }
    }
};

template<typename inType, typename outType>
void copyArray(Array<outType> &dst, Array<inType> const &src) {
    static_assert(!(is_complex<inType>::value && !is_complex<outType>::value),
                  "Cannot copy from complex value to a non complex value");
    copyWrapper<inType, outType> copyFn;
    copyFn(dst, src);
}

#define INSTANTIATE(T)                                        \
    template void copyData<T>(T * data, const Array<T> &src); \
    template Array<T> copyArray<T>(const Array<T> &src);      \
    template void multiply_inplace<T>(Array<T> & src, double norm);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

#define INSTANTIATE_COPY_ARRAY(SRC_T)                                 \
    template void copyArray<SRC_T, float>(Array<float> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, double>(Array<double> & dst,       \
                                           Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, cfloat>(Array<cfloat> & dst,       \
                                           Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> & dst,     \
                                            Array<SRC_T> const &src); \
    template void copyArray<SRC_T, int>(Array<int> & dst,             \
                                        Array<SRC_T> const &src);     \
    template void copyArray<SRC_T, uint>(Array<uint> & dst,           \
                                         Array<SRC_T> const &src);    \
    template void copyArray<SRC_T, intl>(Array<intl> & dst,           \
                                         Array<SRC_T> const &src);    \
    template void copyArray<SRC_T, uintl>(Array<uintl> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, short>(Array<short> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, ushort>(Array<ushort> & dst,       \
                                           Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, uchar>(Array<uchar> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, char>(Array<char> & dst,           \
                                         Array<SRC_T> const &src);    \
    template void copyArray<SRC_T, half>(Array<half> & dst,           \
                                         Array<SRC_T> const &src);

INSTANTIATE_COPY_ARRAY(float)
INSTANTIATE_COPY_ARRAY(double)
INSTANTIATE_COPY_ARRAY(int)
INSTANTIATE_COPY_ARRAY(uint)
INSTANTIATE_COPY_ARRAY(intl)
INSTANTIATE_COPY_ARRAY(uintl)
INSTANTIATE_COPY_ARRAY(uchar)
INSTANTIATE_COPY_ARRAY(char)
INSTANTIATE_COPY_ARRAY(short)
INSTANTIATE_COPY_ARRAY(ushort)
INSTANTIATE_COPY_ARRAY(half)

#define INSTANTIATE_COPY_ARRAY_COMPLEX(SRC_T)                        \
    template void copyArray<SRC_T, cfloat>(Array<cfloat> & dst,      \
                                           Array<SRC_T> const &src); \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> & dst,    \
                                            Array<SRC_T> const &src);

INSTANTIATE_COPY_ARRAY_COMPLEX(cfloat)
INSTANTIATE_COPY_ARRAY_COMPLEX(cdouble)

template<typename T>
T getScalar(const Array<T> &src) {
    T retVal{};
    getQueue().enqueueReadBuffer(
        *src.get(), CL_TRUE, sizeof(T) * src.getOffset(), sizeof(T), &retVal);
    return retVal;
}

#define INSTANTIATE_GETSCALAR(T) template T getScalar(const Array<T> &in);

INSTANTIATE_GETSCALAR(float)
INSTANTIATE_GETSCALAR(double)
INSTANTIATE_GETSCALAR(cfloat)
INSTANTIATE_GETSCALAR(cdouble)
INSTANTIATE_GETSCALAR(int)
INSTANTIATE_GETSCALAR(uint)
INSTANTIATE_GETSCALAR(uchar)
INSTANTIATE_GETSCALAR(char)
INSTANTIATE_GETSCALAR(intl)
INSTANTIATE_GETSCALAR(uintl)
INSTANTIATE_GETSCALAR(short)
INSTANTIATE_GETSCALAR(ushort)
INSTANTIATE_GETSCALAR(half)

}  // namespace opencl
}  // namespace arrayfire
