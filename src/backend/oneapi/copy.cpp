/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <copy.hpp>

#include <Array.hpp>
#include <common/complex.hpp>
#include <common/half.hpp>
#include <err_oneapi.hpp>
#include <kernel/memcopy.hpp>
#include <math.hpp>

using arrayfire::common::half;
using arrayfire::common::is_complex;

using sycl::access_mode;
using sycl::accessor;
using sycl::buffer;
using sycl::id;
using sycl::range;
using sycl::target;

namespace arrayfire {
namespace oneapi {

template<typename T>
void copyData(T *data, const Array<T> &src) {
    if (src.elements() > 0) {
        Array<T> lin = src.isReady() && src.isLinear() ? src : copyArray(src);
        size_t elements = lin.elements();
        Param<T> p      = lin;
        getQueue()
            .submit([&](sycl::handler &h) {
                sycl::range rr(elements);
                sycl::id offset_id(p.info.offset);
                auto offset_acc =
                    p.data->template get_access<sycl::access_mode::read_write>(
                        h, rr, offset_id);
                h.copy(offset_acc, data);
            })
            .wait();
    }
}

template<typename T>
Array<T> copyArray(const Array<T> &A) {
    Array<T> out = createEmptyArray<T>(A.dims());
    if (A.elements() == 0) { return out; }

    dim_t offset = A.getOffset();
    if (A.isReady()) {
        if (A.isLinear()) {
            // FIXME: Add checks

            sycl::buffer<T> *A_buf   = A.get();
            sycl::buffer<T> *out_buf = out.get();

            size_t aelem = A.elements();
            getQueue().submit([&](sycl::handler &h) {
                range rr(aelem);
                id offset_id(offset);
                accessor offset_acc_A =
                    A_buf->template get_access<access_mode::read>(h, rr,
                                                                  offset_id);
                accessor acc_out =
                    out_buf->template get_access<access_mode::write>(h);

                h.copy(offset_acc_A, acc_out);
            });
        } else {
            kernel::memcopy<T>(out.get(), out.strides().get(), A.get(),
                               A.dims().get(), A.strides().get(), offset,
                               (uint)A.ndims());
        }
    } else {
        Param info = {out.get(),
                      {{A.dims().dims[0], A.dims().dims[1], A.dims().dims[2],
                        A.dims().dims[3]},
                       {out.strides().dims[0], out.strides().dims[1],
                        out.strides().dims[2], out.strides().dims[3]},
                       0}};
        evalNodes(info, A.getNode().get());
    }
    return out;
}

template<typename T>
void multiply_inplace(Array<T> &in, double val) {
    kernel::copy<T, T>(in, in, in.ndims(), scalar<T>(0), val, true);
}

template<typename inType, typename outType>
struct copyWrapper {
    void operator()(Array<outType> &out, Array<inType> const &in) {
        kernel::copy<inType, outType>(out, in, in.ndims(), scalar<outType>(0),
                                      1, in.dims() == out.dims());
    }
};

template<typename T>
struct copyWrapper<T, T> {
    void operator()(Array<T> &out, Array<T> const &in) {
        if (out.isLinear() && in.isLinear() &&
            out.elements() == in.elements()) {
            dim_t in_offset  = in.getOffset();
            dim_t out_offset = out.getOffset();

            sycl::buffer<T> *in_buf  = in.get();
            sycl::buffer<T> *out_buf = out.get();

            getQueue()
                .submit([&](sycl::handler &h) {
                    sycl::range rr(in.elements());
                    sycl::id in_offset_id(in_offset);
                    sycl::id out_offset_id(out_offset);

                    auto offset_acc_in =
                        in_buf->template get_access<access_mode::read>(
                            h, rr, in_offset_id);
                    auto offset_acc_out =
                        out_buf->template get_access<access_mode::write>(
                            h, rr, out_offset_id);

                    h.copy(offset_acc_in, offset_acc_out);
                })
                .wait();
        } else {
            kernel::copy<T, T>(out, in, in.ndims(), scalar<T>(0), 1,
                               in.dims() == out.dims());
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

    auto in_get = in.get();
    getQueue()
        .submit([&](sycl::handler &h) {
            auto acc_in =
                in_get->template get_access<sycl::access::mode::read>(
                    h, sycl::range{1},
                    sycl::id{static_cast<uintl>(in.getOffset())});
            h.copy(acc_in, &retVal);
        })
        .wait();

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

}  // namespace oneapi
}  // namespace arrayfire
