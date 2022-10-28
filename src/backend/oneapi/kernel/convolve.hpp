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
#include <debug_oneapi.hpp>
#include <memory.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <af/defines.h>

#include <string>
#include <vector>

namespace oneapi {
namespace kernel {

// below shared MAX_*_LEN's are calculated based on
// a maximum shared memory configuration of 48KB per block
// considering complex types as well
constexpr int MAX_CONV1_FILTER_LEN = 129;
constexpr int MAX_CONV2_FILTER_LEN = 17;
constexpr int MAX_CONV3_FILTER_LEN = 5;

constexpr int MAX_SCONV_FILTER_LEN = 31;

constexpr int THREADS   = 256;
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int CUBE_X    = 8;
constexpr int CUBE_Y    = 8;
constexpr int CUBE_Z    = 4;

template <typename aT>
struct conv_kparam_t {
  sycl::range<3> global{0, 0, 0};
  sycl::range<3> local{0, 0, 0};
  size_t loc_size;
  int nBBS0;
  int nBBS1;
  bool outHasNoOffset;
  bool inHasNoOffset;
  bool launchMoreBlocks;
  int o[3];
  int s[3];
  sycl::buffer<aT>* impulse;
};

template <typename T>
T binOp(T lhs, T rhs) { return lhs * rhs; }

template<typename aT>
void prepareKernelArgs(conv_kparam_t<aT>& param, dim_t* oDims, const dim_t* fDims,
                       const int rank) {
    using sycl::range;

    int batchDims[4] = {1, 1, 1, 1};
    for (int i = rank; i < 4; ++i) {
      batchDims[i] = (param.launchMoreBlocks ? 1 : oDims[i]);
    }

    if (rank == 1) {
      param.local = range<3>{THREADS, 1, 1};
      param.nBBS0 = divup(oDims[0], THREADS);
      param.nBBS1 = batchDims[2];
      param.global = range<3>(
                              param.nBBS0 * THREADS * batchDims[1],
                              param.nBBS1 * batchDims[3],
                              1);
      param.loc_size = (THREADS + 2 * (fDims[0] - 1));
    } else if (rank == 2) {
      param.local = range<3>{THREADS_X, THREADS_Y, 1};
      param.nBBS0 = divup(oDims[0], THREADS_X);
      param.nBBS1 = divup(oDims[1], THREADS_Y);
      param.global = range<3>(
                              param.nBBS0 * THREADS_X * batchDims[2],
                              param.nBBS1 * THREADS_Y * batchDims[3],
                              1);
    } else if (rank == 3) {
      param.local    = range<3>{CUBE_X, CUBE_Y, CUBE_Z};
      param.nBBS0    = divup(oDims[0], CUBE_X);
      param.nBBS1    = divup(oDims[1], CUBE_Y);
      int blk_z      = divup(oDims[2], CUBE_Z);
      param.global = range<3>(param.nBBS0 * CUBE_X * batchDims[3],
                              param.nBBS1 * CUBE_Y, blk_z * CUBE_Z);
      param.loc_size = (CUBE_X + 2 * (fDims[0] - 1)) *
        (CUBE_Y + 2 * (fDims[1] - 1)) *
        (CUBE_Z + 2 * (fDims[2] - 1));
    }
}

template <typename T>
void memcpyBuffer(sycl::buffer<T, 1> &dest, sycl::buffer<T, 1> &src,
                  const size_t n, const size_t srcOffset) {
    getQueue().submit([&](auto &h) {
        sycl::accessor srcAcc{src, h, sycl::range{n}, sycl::id{srcOffset}, sycl::read_only};
        sycl::accessor destAcc{dest, h, sycl::range{n}, sycl::id{0}, sycl::write_only, sycl::no_init};
        h.copy(srcAcc, destAcc);
    });
}


template <typename T, typename aT>
void conv1Helper(const conv_kparam_t<aT> &param, Param<T> &out, const Param<T> &signal,
                 const Param<aT> &filter, const int rank, const bool expand) {
    getQueue().submit([&](auto &h) {
        sycl::accessor<aT, 1, sycl::access::mode::read_write, sycl::access::target::local>
          localMem(param.loc_size, h);
        sycl::accessor outAcc{*out.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor signalAcc{*signal.data, h, sycl::read_only};
        sycl::accessor impulseAcc{*param.impulse, h, sycl::read_only};
        h.parallel_for(sycl::nd_range{param.global, param.local}, [=](sycl::nd_item<3> it) {
            sycl::group g = it.get_group();

            int fLen          = filter.info.dims[0];
            int padding       = fLen - 1;
            int shrdLen       = g.get_local_range(0) + 2 * padding;
            const unsigned b1 = g.get_group_id(0) / param.nBBS0;
            const unsigned b0 = g.get_group_id(0) - param.nBBS0 * b1;
            const unsigned b3 = g.get_group_id(1) / param.nBBS1;
            const unsigned b2 = g.get_group_id(1) - param.nBBS1 * b3;

            T *outDataPtr = outAcc.get_pointer();
            T *dst =
              outDataPtr +
              (b1         * out.info.strides[1] + /* activated with batched input signal */
               param.o[0] * out.info.strides[1] + /* activated with batched input filter */
               b2         * out.info.strides[2] + /* activated with batched input signal */
               param.o[1] * out.info.strides[2] + /* activated with batched input filter */
               b3         * out.info.strides[3] + /* activated with batched input signal */
               param.o[2] * out.info.strides[3]); /* activated with batched input filter */

            const T *signalPtr = signalAcc.get_pointer();
            const T *src =
              signalPtr + signal.info.offset +
              (b1         * signal.info.strides[1] + /* activated with batched input signal */
               param.s[0] * signal.info.strides[1] + /* activated with batched input filter */
               b2         * signal.info.strides[2] + /* activated with batched input signal */
               param.s[1] * signal.info.strides[2] + /* activated with batched input filter */
               b3         * signal.info.strides[3] + /* activated with batched input signal */
               param.s[2] * signal.info.strides[3]); /* activated with batched input filter */

            int gx = g.get_local_range(0) * b0;

            for (int i = it.get_local_id(0); i < shrdLen; i += g.get_local_range(0)) {
                int idx     = gx - padding + i;
                localMem[i] = (idx >= 0 && idx < signal.info.dims[0])
                                  ? src[idx * signal.info.strides[0]]
                                  : (T)(0);
            }
            it.barrier();
            gx += it.get_local_id()[0];

            if (gx >= 0 && gx < out.info.dims[0]) {
              int lx        = g.get_local_id()[0] + padding + ((expand ? 1 : 0) ? 0 : fLen >> 1);
                aT accum = (aT)(0);
                const aT *impulsePtr = impulseAcc.get_pointer();
                for (int f = 0; f < fLen; ++f) {
                  // binOp will do MUL_OP for convolution operation
                  accum = accum + binOp((aT)localMem[lx - f], impulsePtr[f]);
                }
                dst[gx] = (T)accum;
            }
        });
    });
}

int index(int i, int j, int k, int jstride, int kstride) {
    return i + j * jstride + k * kstride;
}

template <typename T, typename aT>
void conv3Helper(const conv_kparam_t<aT> &param, Param<T> &out,
                 const Param<T> &signal, const Param<aT> &filter,
                 const int rank, const bool expand) {
    getQueue().submit([&](auto &h) {
        sycl::accessor<aT, 1, sycl::access::mode::read_write, sycl::access::target::local>
          localMem(param.loc_size, h);
        sycl::accessor outAcc{*out.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor signalAcc{*signal.data, h, sycl::read_only};
        sycl::accessor impulseAcc{*param.impulse, h, sycl::read_only};
        h.parallel_for(sycl::nd_range{param.global, param.local}, [=](sycl::nd_item<3> it) {
            sycl::group g = it.get_group();

            int fLen0    = filter.info.dims[0];
            int fLen1    = filter.info.dims[1];
            int fLen2    = filter.info.dims[2];
            int radius0  = fLen0 - 1;
            int radius1  = fLen1 - 1;
            int radius2  = fLen2 - 1;
            int shrdLen0 = g.get_local_range(0) + 2 * radius0;
            int shrdLen1 = g.get_local_range(1) + 2 * radius1;
            int shrdLen2 = g.get_local_range(2) + 2 * radius2;
            int skStride = shrdLen0 * shrdLen1;
            int fStride  = fLen0 * fLen1;
            unsigned b2  = g.get_group_id(0) / param.nBBS0;

            T *outDataPtr = outAcc.get_pointer();
            T *dst =
              outDataPtr +
              (b2         * out.info.strides[3] + /* activated with batched input signal */
               param.o[2] * out.info.strides[3]); /* activated with batched input filter */

            const T *signalPtr = signalAcc.get_pointer();
            const T *src =
              signalPtr + signal.info.offset +
              (b2         * signal.info.strides[3] + /* activated with batched input signal */
               param.s[2] * signal.info.strides[3]); /* activated with batched input filter */

            int lx  = it.get_local_id(0);
            int ly  = it.get_local_id(1);
            int lz  = it.get_local_id(2);
            int gx  = g.get_local_range(0) * (g.get_group_id(0) - b2 * param.nBBS0) + lx;
            int gy  = g.get_local_range(1) * g.get_group_id(1) + ly;
            int gz  = g.get_local_range(2) * g.get_group_id(2) + lz;
            int lx2 = lx + g.get_local_range(0);
            int ly2 = ly + g.get_local_range(1);
            int lz2 = lz + g.get_local_range(2);
            int gx2 = gx + g.get_local_range(0);
            int gy2 = gy + g.get_local_range(1);
            int gz2 = gz + g.get_local_range(2);

            int s0 = signal.info.strides[0];
            int s1 = signal.info.strides[1];
            int s2 = signal.info.strides[2];
            int d0 = signal.info.dims[0];
            int d1 = signal.info.dims[1];
            int d2 = signal.info.dims[2];

            for (int c = lz, gz2 = gz; c < shrdLen2;
                 c += g.get_local_range(2), gz2 += g.get_local_range(2)) {
              int k     = gz2 - radius2;
              bool is_k = k >= 0 && k < d2;
              for (int b = ly, gy2 = gy; b < shrdLen1;
                   b += g.get_local_range(1), gy2 += g.get_local_range(1)) {
                int j     = gy2 - radius1;
                bool is_j = j >= 0 && j < d1;
                for (int a = lx, gx2 = gx; a < shrdLen0;
                     a += g.get_local_range(0), gx2 += g.get_local_range(0)) {
                  int i     = gx2 - radius0;
                  bool is_i = i >= 0 && i < d0;
                  localMem[c * skStride + b * shrdLen0 + a] =
                    (is_i && is_j && is_k ? src[i * s0 + j * s1 + k * s2]
                     : (T)(0));
                }
              }
            }
            it.barrier();

            if (gx < out.info.dims[0] && gy < out.info.dims[1] && gz < out.info.dims[2]) {
              int ci = lx + radius0 + ((expand ? 1 : 0) ? 0 : fLen0 >> 1);
              int cj = ly + radius1 + ((expand ? 1 : 0) ? 0 : fLen1 >> 1);
              int ck = lz + radius2 + ((expand ? 1 : 0) ? 0 : fLen2 >> 1);

              aT accum = (aT)(0);
              for (int fk = 0; fk < fLen2; ++fk) {
                for (int fj = 0; fj < fLen1; ++fj) {
                  for (int fi = 0; fi < fLen0; ++fi) {
                    aT f_val = impulseAcc[index(fi, fj, fk, fLen0, fStride)];
                    T s_val = localMem[index(ci - fi, cj - fj, ck - fk,
                                             shrdLen0, skStride)];

                    // binOp will do MUL_OP for convolution operation
                    accum = accum + binOp((aT)s_val, (aT)f_val);
                  }
                }
              }
              dst[index(gx, gy, gz, out.info.strides[1], out.info.strides[2])] =
                (T)accum;
            }
        });
    });
}

template<typename T, typename aT>
void conv3(conv_kparam_t<aT>& p, Param<T>& out, const Param<T>& sig, const Param<aT>& filt,
           const bool expand) {
    size_t se_size = filt.info.dims[0] * filt.info.dims[1] * filt.info.dims[2];
    sycl::buffer<aT> impulse{sycl::range(se_size)};
    int f0Off = filt.info.offset;

    for (int b3 = 0; b3 < filt.info.dims[3]; ++b3) {
        int f3Off = b3 * filt.info.strides[3];

        const size_t srcOffset = f3Off + f0Off;
        memcpyBuffer(impulse, *filt.data, se_size, srcOffset);
        p.impulse = &impulse;

        p.o[2] = (p.outHasNoOffset ? 0 : b3);
        p.s[2] = (p.inHasNoOffset ? 0 : b3);

        conv3Helper<T, aT>(p, out, sig, filt, 3, expand);
    }
}

#define INSTANTIATE_CONV3(T, aT)                                        \
  template void conv3<T, aT>(conv_kparam_t<aT>&, Param<T>&, const Param<T>&, \
                             const Param<aT>&, const bool);

INSTANTIATE_CONV3(cdouble, cdouble)
INSTANTIATE_CONV3(cfloat, cfloat)
INSTANTIATE_CONV3(double, double)
INSTANTIATE_CONV3(float, float)
INSTANTIATE_CONV3(uint, float)
INSTANTIATE_CONV3(int, float)
INSTANTIATE_CONV3(uchar, float)
INSTANTIATE_CONV3(char, float)
INSTANTIATE_CONV3(ushort, float)
INSTANTIATE_CONV3(short, float)
INSTANTIATE_CONV3(uintl, float)
INSTANTIATE_CONV3(intl, float)


template<typename T, typename aT>
void conv2Helper(const conv_kparam_t<aT>& param, Param<T> out, const Param<T> signal,
                 const Param<aT> filter, const bool expand) {
    constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    const int f0 = filter.info.dims[0];
    const int f1 = filter.info.dims[1];
    const size_t LOC_SIZE =
        (THREADS_X + 2 * (f0 - 1)) * (THREADS_Y + 2 * (f1 - 1));

    getQueue().submit([&](auto &h) {
        sycl::accessor<aT, 1, sycl::access::mode::read_write, sycl::access::target::local>
          localMem(LOC_SIZE, h);
        sycl::accessor outAcc{*out.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor signalAcc{*signal.data, h, sycl::read_only};
        sycl::accessor impulseAcc{*param.impulse, h, sycl::read_only};
        h.parallel_for(sycl::nd_range{param.global, param.local}, [=](sycl::nd_item<3> it) {
            sycl::group g = it.get_group();

            int radius0  = f0 - 1;
            int radius1  = f1 - 1;
            int padding0 = 2 * radius0;
            int padding1 = 2 * radius1;
            int shrdLen0 = g.get_local_range(0) + padding0;
            int shrdLen1 = g.get_local_range(1) + padding1;

            unsigned b0 = g.get_group_id(0) / param.nBBS0;
            unsigned b1 = g.get_group_id(1) / param.nBBS1;

            T *outDataPtr = outAcc.get_pointer();
            T *dst =
              outDataPtr +
              (b0         * out.info.strides[2] + /* activated with batched input signal */
               param.o[1] * out.info.strides[2] + /* activated with batched input filter */
               b1         * out.info.strides[3] + /* activated with batched input signal */
               param.o[2] * out.info.strides[3]); /* activated with batched input filter */

            const T *signalPtr = signalAcc.get_pointer();
            const T *src =
              signalPtr + signal.info.offset +
              (b0         * signal.info.strides[2] + /* activated with batched input signal */
               param.s[1] * signal.info.strides[2] + /* activated with batched input filter */
               b1         * signal.info.strides[3] + /* activated with batched input signal */
               param.s[2] * signal.info.strides[3]); /* activated with batched input filter */

            int lx = it.get_local_id(0);
            int ly = it.get_local_id(1);
            int gx = g.get_local_range(0) * (g.get_group_id(0) - b0 * param.nBBS0) + lx;
            int gy = g.get_local_range(1) * (g.get_group_id(1) - b1 * param.nBBS1) + ly;

            // below loops are traditional loops, they only run multiple
            // times filter length is more than launch size
            int s0 = signal.info.strides[0];
            int s1 = signal.info.strides[1];
            int d0 = signal.info.dims[0];
            int d1 = signal.info.dims[1];
            for (int b = ly, gy2 = gy; b < shrdLen1;
                 b += g.get_local_range(1), gy2 += g.get_local_range(1)) {
              int j     = gy2 - radius1;
              bool is_j = j >= 0 && j < d1;
              // move row_set get_local_size(1) along coloumns
              for (int a = lx, gx2 = gx; a < shrdLen0;
                   a += g.get_local_range(0), gx2 += g.get_local_range(0)) {
                int i     = gx2 - radius0;
                bool is_i = i >= 0 && i < d0;
                localMem[b * shrdLen0 + a] =
                  (is_i && is_j ? src[i * s0 + j * s1] : (T)(0));
              }
            }
            it.barrier();

            if (gx < out.info.dims[0] && gy < out.info.dims[1]) {
              int ci = lx + radius0 + ((expand ? 1 : 0) ? 0 : f0 >> 1);
              int cj = ly + radius1 + ((expand ? 1 : 0) ? 0 : f1 >> 1);

              aT accum = (aT)(0);
              for (int fj = 0; fj < f1; ++fj) {
                for (int fi = 0; fi < f0; ++fi) {
                  aT f_val = impulseAcc[fj * f0 + fi];
                  T s_val = localMem[(cj - fj) * shrdLen0 + (ci - fi)];

                  // binOp will do MUL_OP for convolution operation
                  accum = accum + binOp((aT)s_val, (aT)f_val);
                }
              }
              dst[gy * out.info.strides[1] + gx] = (T)accum;
            }
        });
    });
}

template<typename T, typename aT>
void conv2(conv_kparam_t<aT>& p, Param<T>& out, const Param<T>& sig, const Param<aT>& filt,
           const bool expand) {
    size_t se_size = filt.info.dims[0] * filt.info.dims[1];
    sycl::buffer<aT> impulse{sycl::range(se_size)};
    int f0Off      = filt.info.offset;

    for (int b3 = 0; b3 < filt.info.dims[3]; ++b3) {
        int f3Off = b3 * filt.info.strides[3];

        for (int b2 = 0; b2 < filt.info.dims[2]; ++b2) {
            int f2Off = b2 * filt.info.strides[2];

            const size_t srcOffset = f2Off + f3Off + f0Off;
            memcpyBuffer(impulse, *filt.data, se_size, srcOffset);
            p.impulse = &impulse;

            p.o[1] = (p.outHasNoOffset ? 0 : b2);
            p.o[2] = (p.outHasNoOffset ? 0 : b3);
            p.s[1] = (p.inHasNoOffset ? 0 : b2);
            p.s[2] = (p.inHasNoOffset ? 0 : b3);

            conv2Helper<T, aT>(p, out, sig, filt, expand);
        }
    }
}

#define INSTANTIATE_CONV2(T, aT)                                      \
 template void conv2<T, aT>(conv_kparam_t<aT>&, Param<T>&, const Param<T>&,    \
                               const Param<aT>&, const bool);

INSTANTIATE_CONV2(char, float)
INSTANTIATE_CONV2(cfloat, cfloat)
INSTANTIATE_CONV2(cdouble, cdouble)
INSTANTIATE_CONV2(float, float)
INSTANTIATE_CONV2(double, double)
INSTANTIATE_CONV2(short, float)
INSTANTIATE_CONV2(int, float)
INSTANTIATE_CONV2(intl, float)
INSTANTIATE_CONV2(ushort, float)
INSTANTIATE_CONV2(uint, float)
INSTANTIATE_CONV2(uintl, float)
INSTANTIATE_CONV2(uchar, float)


template<typename T, typename aT>
void conv1(conv_kparam_t<aT>& p, Param<T>& out, const Param<T>& sig, const Param<aT>& filt,
           const bool expand) {
    const size_t se_size = filt.info.dims[0];
    sycl::buffer<aT> impulse{sycl::range(filt.info.dims[0])};
    int f0Off = filt.info.offset;
    for (int b3 = 0; b3 < filt.info.dims[3]; ++b3) {
        int f3Off = b3 * filt.info.strides[3];

        for (int b2 = 0; b2 < filt.info.dims[2]; ++b2) {
            int f2Off = b2 * filt.info.strides[2];

            for (int b1 = 0; b1 < filt.info.dims[1]; ++b1) {
                int f1Off = b1 * filt.info.strides[1];

                const size_t srcOffset = f0Off + f1Off + f2Off + f3Off;
                memcpyBuffer(impulse, *filt.data, se_size, srcOffset);
                p.impulse = &impulse;

                p.o[0] = (p.outHasNoOffset ? 0 : b1);
                p.o[1] = (p.outHasNoOffset ? 0 : b2);
                p.o[2] = (p.outHasNoOffset ? 0 : b3);
                p.s[0] = (p.inHasNoOffset ? 0 : b1);
                p.s[1] = (p.inHasNoOffset ? 0 : b2);
                p.s[2] = (p.inHasNoOffset ? 0 : b3);

                conv1Helper<T, aT>(p, out, sig, filt, 1, expand);
            }
        }
    }
}

#define INSTANTIATE_CONV1(T, aT)                                        \
  template void conv1<T, aT>(conv_kparam_t<aT>&, Param<T>&, const Param<T>&, const Param<aT>&, const bool);

INSTANTIATE_CONV1(cdouble, cdouble)
INSTANTIATE_CONV1(cfloat, cfloat)
INSTANTIATE_CONV1(double, double)
INSTANTIATE_CONV1(float, float)
INSTANTIATE_CONV1(uint, float)
INSTANTIATE_CONV1(int, float)
INSTANTIATE_CONV1(uchar, float)
INSTANTIATE_CONV1(char, float)
INSTANTIATE_CONV1(ushort, float)
INSTANTIATE_CONV1(short, float)
INSTANTIATE_CONV1(uintl, float)
INSTANTIATE_CONV1(intl, float)

template<typename T, typename aT>
void convolve_nd(Param<T> out, const Param<T> signal, const Param<aT> filter,
                 AF_BATCH_KIND kind, const int rank, const bool expand) {
    conv_kparam_t<aT> param;

    for (int i = 0; i < 3; ++i) {
        param.o[i] = 0;
        param.s[i] = 0;
    }
    param.launchMoreBlocks = kind == AF_BATCH_SAME || kind == AF_BATCH_RHS;
    param.outHasNoOffset   = kind == AF_BATCH_LHS || kind == AF_BATCH_NONE;
    param.inHasNoOffset    = kind != AF_BATCH_SAME;

    prepareKernelArgs<aT>(param, out.info.dims, filter.info.dims, rank);

    switch (rank) {
    case 1: conv1<T, aT>(param, out, signal, filter, expand); break;
    case 2: conv2<T, aT>(param, out, signal, filter, expand); break;
    case 3: conv3<T, aT>(param, out, signal, filter, expand); break;
    }

    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
