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
#include <debug_oneapi.hpp>
#include <memory.hpp>
#include <traits.hpp>

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

template<typename aT>
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
    sycl::buffer<aT> *impulse;
};

template<typename T>
T binOp(T lhs, T rhs) {
    return lhs * rhs;
}

template<typename aT>
void prepareKernelArgs(conv_kparam_t<aT> &param, dim_t *oDims,
                       const dim_t *fDims, const int rank) {
    using sycl::range;

    int batchDims[4] = {1, 1, 1, 1};
    for (int i = rank; i < 4; ++i) {
        batchDims[i] = (param.launchMoreBlocks ? 1 : oDims[i]);
    }

    if (rank == 1) {
        param.local    = range<3>{THREADS, 1, 1};
        param.nBBS0    = divup(oDims[0], THREADS);
        param.nBBS1    = batchDims[2];
        param.global   = range<3>(param.nBBS0 * THREADS * batchDims[1],
                                param.nBBS1 * batchDims[3], 1);
        param.loc_size = (THREADS + 2 * (fDims[0] - 1));
    } else if (rank == 2) {
        param.local  = range<3>{THREADS_X, THREADS_Y, 1};
        param.nBBS0  = divup(oDims[0], THREADS_X);
        param.nBBS1  = divup(oDims[1], THREADS_Y);
        param.global = range<3>(param.nBBS0 * THREADS_X * batchDims[2],
                                param.nBBS1 * THREADS_Y * batchDims[3], 1);
    } else if (rank == 3) {
        param.local    = range<3>{CUBE_X, CUBE_Y, CUBE_Z};
        param.nBBS0    = divup(oDims[0], CUBE_X);
        param.nBBS1    = divup(oDims[1], CUBE_Y);
        int blk_z      = divup(oDims[2], CUBE_Z);
        param.global   = range<3>(param.nBBS0 * CUBE_X * batchDims[3],
                                param.nBBS1 * CUBE_Y, blk_z * CUBE_Z);
        param.loc_size = (CUBE_X + 2 * (fDims[0] - 1)) *
                         (CUBE_Y + 2 * (fDims[1] - 1)) *
                         (CUBE_Z + 2 * (fDims[2] - 1));
    }
}

template<typename T>
void memcpyBuffer(sycl::buffer<T, 1> &dest, sycl::buffer<T, 1> &src,
                  const size_t n, const size_t srcOffset) {
    getQueue().submit([&](auto &h) {
        sycl::accessor srcAcc{src, h, sycl::range{n}, sycl::id{srcOffset},
                              sycl::read_only};
        sycl::accessor destAcc{
            dest,         h, sycl::range{n}, sycl::id{0}, sycl::write_only,
            sycl::no_init};
        h.copy(srcAcc, destAcc);
    });
}

template<typename T, typename aT>
void convSep(Param<T> out, const Param<T> signal, const Param<aT> filter,
             const int conv_dim, const bool expand) {
    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;
    constexpr bool IsComplex =
        std::is_same<T, cfloat>::value || std::is_same<T, cdouble>::value;

    const int fLen       = filter.info.dims[0] * filter.info.dims[1];
    const size_t C0_SIZE = (THREADS_X + 2 * (fLen - 1)) * THREADS_Y;
    const size_t C1_SIZE = (THREADS_Y + 2 * (fLen - 1)) * THREADS_X;
    size_t locSize       = (conv_dim == 0 ? C0_SIZE : C1_SIZE);

    sycl::range<2> local{THREADS_X, THREADS_Y};

    int blk_x = divup(out.info.dims[0], THREADS_X);
    int blk_y = divup(out.info.dims[1], THREADS_Y);

    sycl::range<2> global(blk_x * signal.info.dims[2] * THREADS_X,
                          blk_y * signal.info.dims[3] * THREADS_Y);

    sycl::buffer<aT> mBuff{sycl::range(fLen)};
    memcpyBuffer(mBuff, *filter.data, fLen, 0);

    getQueue().submit([&](auto &h) {
        sycl::accessor<aT, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            localMem(locSize, h);
        sycl::accessor outAcc{*out.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor signalAcc{*signal.data, h, sycl::read_only};
        sycl::accessor impulseAcc{mBuff, h, sycl::read_only};
        h.parallel_for(sycl::nd_range{global, local}, [=](sycl::nd_item<2> it) {
            sycl::group g = it.get_group();

            const int radius  = fLen - 1;
            const int padding = 2 * radius;
            const int s0      = signal.info.strides[0];
            const int s1      = signal.info.strides[1];
            const int d0      = signal.info.dims[0];
            const int d1      = signal.info.dims[1];
            const int shrdLen =
                g.get_local_range(0) + (conv_dim == 0 ? padding : 0);

            unsigned b2 = g.get_group_id(0) / blk_x;
            unsigned b3 = g.get_group_id(1) / blk_y;

            T *outDataPtr = outAcc.get_pointer();
            T *dst        = outDataPtr +
                     (b2 * out.info.strides[2] + b3 * out.info.strides[3]);

            const T *signalPtr = signalAcc.get_pointer();
            const T *src =
                signalPtr +
                (b2 * signal.info.strides[2] + b3 * signal.info.strides[3]) +
                signal.info.offset;

            int lx = it.get_local_id(0);
            int ly = it.get_local_id(1);
            int ox =
                g.get_local_range(0) * (g.get_group_id(0) - b2 * blk_x) + lx;
            int oy =
                g.get_local_range(1) * (g.get_group_id(1) - b3 * blk_y) + ly;
            int gx = ox;
            int gy = oy;

            // below if-else statement is based on MACRO value passed while
            // kernel compilation
            if (conv_dim == 0) {
                gx += ((expand ? 1 : 0) ? 0 : fLen >> 1);
                int endX = ((fLen - 1) << 1) + g.get_local_range(0);
#pragma unroll
                for (int lx = it.get_local_id(0), glb_x = gx; lx < endX;
                     lx += g.get_local_range(0),
                         glb_x += g.get_local_range(0)) {
                    int i     = glb_x - radius;
                    int j     = gy;
                    bool is_i = i >= 0 && i < d0;
                    bool is_j = j >= 0 && j < d1;
                    localMem[ly * shrdLen + lx] =
                        (is_i && is_j ? src[i * s0 + j * s1] : (T)(0));
                }
            } else if (conv_dim == 1) {
                gy += ((expand ? 1 : 0) ? 0 : fLen >> 1);
                int endY = ((fLen - 1) << 1) + g.get_local_range(1);
#pragma unroll
                for (int ly = it.get_local_id(1), glb_y = gy; ly < endY;
                     ly += it.get_local_range(1),
                         glb_y += g.get_local_range(1)) {
                    int i     = gx;
                    int j     = glb_y - radius;
                    bool is_i = i >= 0 && i < d0;
                    bool is_j = j >= 0 && j < d1;
                    localMem[ly * shrdLen + lx] =
                        (is_i && is_j ? src[i * s0 + j * s1] : (T)(0));
                }
            }
            it.barrier();

            if (ox < out.info.dims[0] && oy < out.info.dims[1]) {
                // below conditional statement is based on MACRO value passed
                // while kernel compilation
                int i    = (conv_dim == 0 ? lx : ly) + radius;
                aT accum = (aT)(0);
#pragma unroll
                for (int f = 0; f < fLen; ++f) {
                    aT f_val = impulseAcc[f];

                    // below conditional statement is based on MACRO value
                    // passed while kernel compilation
                    int s_idx = (conv_dim == 0 ? (ly * shrdLen + (i - f))
                                               : ((i - f) * shrdLen + lx));
                    T s_val   = localMem[s_idx];

                    // binOp will do MUL_OP for convolution operation
                    accum = accum + binOp((aT)s_val, (aT)f_val);
                }
                dst[oy * out.info.strides[1] + ox] = (T)accum;
            }
        });
    });
}

#define INSTANTIATE_SEPARABLE(T, accT)                                \
    template void convSep<T, accT>(Param<T>, const Param<T>,          \
                                   const Param<accT> filt, const int, \
                                   const bool);

INSTANTIATE_SEPARABLE(cdouble, cdouble)
INSTANTIATE_SEPARABLE(cfloat, cfloat)
INSTANTIATE_SEPARABLE(double, double)
INSTANTIATE_SEPARABLE(float, float)
INSTANTIATE_SEPARABLE(uint, float)
INSTANTIATE_SEPARABLE(int, float)
INSTANTIATE_SEPARABLE(uchar, float)
INSTANTIATE_SEPARABLE(char, float)
INSTANTIATE_SEPARABLE(ushort, float)
INSTANTIATE_SEPARABLE(short, float)
INSTANTIATE_SEPARABLE(uintl, float)
INSTANTIATE_SEPARABLE(intl, float)

}  // namespace kernel
}  // namespace oneapi
