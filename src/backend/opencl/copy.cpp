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

using common::half;
using common::is_complex;

namespace opencl {

template<typename T>
void copyData(T *data, const Array<T> &A) {
    // FIXME: Merge this with copyArray
    A.eval();

    dim_t offset = 0;
    cl::Buffer buf;
    Array<T> out = A;

    if (A.isLinear() ||  // No offsets, No strides
        A.ndims() == 1   // Simple offset, no strides.
    ) {
        buf    = *A.get();
        offset = A.getOffset();
    } else {
        // FIXME: Think about implementing eval
        out    = copyArray(A);
        buf    = *out.get();
        offset = 0;
    }

    // FIXME: Add checks
    getQueue().enqueueReadBuffer(buf, CL_TRUE, sizeof(T) * offset,
                                 sizeof(T) * A.elements(), data);
}

template<typename T>
Array<T> copyArray(const Array<T> &A) {
    Array<T> out = createEmptyArray<T>(A.dims());
    dim_t offset = A.getOffset();

    if (A.isLinear()) {
        // FIXME: Add checks
        getQueue().enqueueCopyBuffer(*A.get(), *out.get(), sizeof(T) * offset,
                                     0, A.elements() * sizeof(T));
    } else {
        kernel::memcopy<T>(*out.get(), out.strides().get(), *A.get(),
                           A.dims().get(), A.strides().get(), offset,
                           (uint)A.ndims());
    }
    return out;
}

template<typename T>
void multiply_inplace(Array<T> &in, double val) {
    kernel::copy<T, T, true>(in, in, in.ndims(), scalar<T>(0), val);
}

template<typename inType, typename outType>
struct copyWrapper {
    void operator()(Array<outType> &out, Array<inType> const &in) {
        if (in.dims() == out.dims()) {
            kernel::copy<inType, outType, true>(out, in, in.ndims(),
                                                scalar<outType>(0), 1);
        } else {
            kernel::copy<inType, outType, false>(out, in, in.ndims(),
                                                 scalar<outType>(0), 1);
        }
    }
};

template<typename T>
struct copyWrapper<T, T> {
    void operator()(Array<T> &out, Array<T> const &in) {
        if (out.isLinear() && in.isLinear() &&
            out.elements() == in.elements()) {
            dim_t in_offset  = in.getOffset() * sizeof(T);
            dim_t out_offset = out.getOffset() * sizeof(T);

            getQueue().enqueueCopyBuffer(*in.get(), *out.get(), in_offset,
                                         out_offset, in.elements() * sizeof(T));
        } else {
            if (in.dims() == out.dims()) {
                kernel::copy<T, T, true>(out, in, in.ndims(), scalar<T>(0), 1);
            } else {
                kernel::copy<T, T, false>(out, in, in.ndims(), scalar<T>(0), 1);
            }
        }
    }
};

template<typename inType, typename outType>
void copyArray(Array<outType> &out, Array<inType> const &in) {
    static_assert(!(is_complex<inType>::value && !is_complex<outType>::value),
                  "Cannot copy from complex value to a non complex value");
    copyWrapper<inType, outType> copyFn;
    copyFn(out, in);
}

#define INSTANTIATE(T)                                         \
    template void copyData<T>(T * data, const Array<T> &from); \
    template Array<T> copyArray<T>(const Array<T> &A);         \
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
T getScalar(const Array<T> &in) {
    T retVal{};
    getQueue().enqueueReadBuffer(*in.get(), CL_TRUE, sizeof(T) * in.getOffset(),
                                 sizeof(T), &retVal);
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
