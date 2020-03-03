/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/complex.hpp>
#include <common/half.hpp>
#include <copy.hpp>
#include <cuda_runtime_api.h>
#include <err_cuda.hpp>
#include <kernel/memcopy.hpp>
#include <math.hpp>

using common::half;
using common::is_complex;

namespace cuda {

template<typename T>
void copyData(T *dst, const Array<T> &src) {
    // FIXME: Merge this with copyArray
    src.eval();

    Array<T> out = src;
    const T *ptr = NULL;

    if (src.isLinear() ||  // No offsets, No strides
        src.ndims() == 1   // Simple offset, no strides.
    ) {
        // A.get() gets data with offsets
        ptr = src.get();
    } else {
        // FIXME: Think about implementing eval
        out = copyArray(src);
        ptr = out.get();
    }

    auto stream = cuda::getActiveStream();
    CUDA_CHECK(cudaMemcpyAsync(dst, ptr, src.elements() * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return;
}

template<typename T>
Array<T> copyArray(const Array<T> &src) {
    Array<T> out = createEmptyArray<T>(src.dims());

    if (src.isLinear()) {
        CUDA_CHECK(
            cudaMemcpyAsync(out.get(), src.get(), src.elements() * sizeof(T),
                            cudaMemcpyDeviceToDevice, cuda::getActiveStream()));
    } else {
        kernel::memcopy<T>(out, src, src.ndims());
    }
    return out;
}

template<typename inType, typename outType>
Array<outType> padArray(Array<inType> const &in, dim4 const &dims,
                        outType default_value, double factor) {
    ARG_ASSERT(1, (in.ndims() == (size_t)dims.ndims()));
    Array<outType> ret = createEmptyArray<outType>(dims);
    kernel::copy<inType, outType>(ret, in, in.ndims(), default_value, factor);
    return ret;
}

template<typename T>
void multiply_inplace(Array<T> &in, double val) {
    kernel::copy<T, T>(in, in, in.ndims(), scalar<T>(0), val);
}

template<typename inType, typename outType>
struct copyWrapper {
    void operator()(Array<outType> &out, Array<inType> const &in) {
        kernel::copy<inType, outType>(out, in, in.ndims(), scalar<outType>(0),
                                      1);
    }
};

template<typename T>
struct copyWrapper<T, T> {
    void operator()(Array<T> &out, Array<T> const &in) {
        if (out.isLinear() && in.isLinear() &&
            out.elements() == in.elements()) {
            CUDA_CHECK(cudaMemcpyAsync(
                out.get(), in.get(), in.elements() * sizeof(T),
                cudaMemcpyDeviceToDevice, cuda::getActiveStream()));
        } else {
            kernel::copy<T, T>(out, in, in.ndims(), scalar<T>(0), 1);
        }
    }
};

template<typename inType, typename outType>
void copyArray(Array<outType> &out, Array<inType> const &in) {
    static_assert(!(is_complex<inType>::value && !is_complex<outType>::value),
                  "Cannot copy from complex value to a non complex value");
    ARG_ASSERT(1, (in.ndims() == (size_t)out.dims().ndims()));
    copyWrapper<inType, outType> copyFn;
    copyFn(out, in);
}

#define INSTANTIATE(T)                                       \
    template void copyData<T>(T * dst, const Array<T> &src); \
    template Array<T> copyArray<T>(const Array<T> &src);     \
    template void multiply_inplace<T>(Array<T> & in, double norm);

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

#define INSTANTIATE_PAD_ARRAY(SRC_T)                                      \
    template Array<float> padArray<SRC_T, float>(                         \
        Array<SRC_T> const &src, dim4 const &dims, float default_value,   \
        double factor);                                                   \
    template Array<double> padArray<SRC_T, double>(                       \
        Array<SRC_T> const &src, dim4 const &dims, double default_value,  \
        double factor);                                                   \
    template Array<cfloat> padArray<SRC_T, cfloat>(                       \
        Array<SRC_T> const &src, dim4 const &dims, cfloat default_value,  \
        double factor);                                                   \
    template Array<cdouble> padArray<SRC_T, cdouble>(                     \
        Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, \
        double factor);                                                   \
    template Array<int> padArray<SRC_T, int>(                             \
        Array<SRC_T> const &src, dim4 const &dims, int default_value,     \
        double factor);                                                   \
    template Array<uint> padArray<SRC_T, uint>(                           \
        Array<SRC_T> const &src, dim4 const &dims, uint default_value,    \
        double factor);                                                   \
    template Array<intl> padArray<SRC_T, intl>(                           \
        Array<SRC_T> const &src, dim4 const &dims, intl default_value,    \
        double factor);                                                   \
    template Array<uintl> padArray<SRC_T, uintl>(                         \
        Array<SRC_T> const &src, dim4 const &dims, uintl default_value,   \
        double factor);                                                   \
    template Array<short> padArray<SRC_T, short>(                         \
        Array<SRC_T> const &src, dim4 const &dims, short default_value,   \
        double factor);                                                   \
    template Array<ushort> padArray<SRC_T, ushort>(                       \
        Array<SRC_T> const &src, dim4 const &dims, ushort default_value,  \
        double factor);                                                   \
    template Array<uchar> padArray<SRC_T, uchar>(                         \
        Array<SRC_T> const &src, dim4 const &dims, uchar default_value,   \
        double factor);                                                   \
    template Array<char> padArray<SRC_T, char>(                           \
        Array<SRC_T> const &src, dim4 const &dims, char default_value,    \
        double factor);                                                   \
    template Array<half> padArray<SRC_T, half>(                           \
        Array<SRC_T> const &src, dim4 const &dims, half default_value,    \
        double factor);                                                   \
    template void copyArray<SRC_T, float>(Array<float> & dst,             \
                                          Array<SRC_T> const &src);       \
    template void copyArray<SRC_T, double>(Array<double> & dst,           \
                                           Array<SRC_T> const &src);      \
    template void copyArray<SRC_T, cfloat>(Array<cfloat> & dst,           \
                                           Array<SRC_T> const &src);      \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> & dst,         \
                                            Array<SRC_T> const &src);     \
    template void copyArray<SRC_T, int>(Array<int> & dst,                 \
                                        Array<SRC_T> const &src);         \
    template void copyArray<SRC_T, uint>(Array<uint> & dst,               \
                                         Array<SRC_T> const &src);        \
    template void copyArray<SRC_T, intl>(Array<intl> & dst,               \
                                         Array<SRC_T> const &src);        \
    template void copyArray<SRC_T, uintl>(Array<uintl> & dst,             \
                                          Array<SRC_T> const &src);       \
    template void copyArray<SRC_T, short>(Array<short> & dst,             \
                                          Array<SRC_T> const &src);       \
    template void copyArray<SRC_T, ushort>(Array<ushort> & dst,           \
                                           Array<SRC_T> const &src);      \
    template void copyArray<SRC_T, uchar>(Array<uchar> & dst,             \
                                          Array<SRC_T> const &src);       \
    template void copyArray<SRC_T, char>(Array<char> & dst,               \
                                         Array<SRC_T> const &src);        \
    template void copyArray<SRC_T, half>(Array<half> & dst,               \
                                         Array<SRC_T> const &src);

INSTANTIATE_PAD_ARRAY(float)
INSTANTIATE_PAD_ARRAY(double)
INSTANTIATE_PAD_ARRAY(int)
INSTANTIATE_PAD_ARRAY(uint)
INSTANTIATE_PAD_ARRAY(intl)
INSTANTIATE_PAD_ARRAY(uintl)
INSTANTIATE_PAD_ARRAY(short)
INSTANTIATE_PAD_ARRAY(ushort)
INSTANTIATE_PAD_ARRAY(uchar)
INSTANTIATE_PAD_ARRAY(char)
INSTANTIATE_PAD_ARRAY(half)

#define INSTANTIATE_PAD_ARRAY_COMPLEX(SRC_T)                              \
    template Array<cfloat> padArray<SRC_T, cfloat>(                       \
        Array<SRC_T> const &src, dim4 const &dims, cfloat default_value,  \
        double factor);                                                   \
    template Array<cdouble> padArray<SRC_T, cdouble>(                     \
        Array<SRC_T> const &src, dim4 const &dims, cdouble default_value, \
        double factor);                                                   \
    template void copyArray<SRC_T, cfloat>(Array<cfloat> & dst,           \
                                           Array<SRC_T> const &src);      \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> & dst,         \
                                            Array<SRC_T> const &src);

INSTANTIATE_PAD_ARRAY_COMPLEX(cfloat)
INSTANTIATE_PAD_ARRAY_COMPLEX(cdouble)

template<typename T>
T getScalar(const Array<T> &in) {
    T retVal;
    CUDA_CHECK(cudaMemcpyAsync(&retVal, in.get(), sizeof(T),
                               cudaMemcpyDeviceToHost,
                               cuda::getActiveStream()));
    CUDA_CHECK(cudaStreamSynchronize(cuda::getActiveStream()));
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

}  // namespace cuda
