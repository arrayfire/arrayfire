/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_oneapi.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
using read_accessor = sycl::accessor<T, 1, sycl::access::mode::read>;
template<typename T>
using write_accessor = sycl::accessor<T, 1, sycl::access::mode::write>;

template<typename T, typename accType>
class convolveSeparableCreateKernel {
   public:
    convolveSeparableCreateKernel(write_accessor<T> out, KParam oInfo,
                                  read_accessor<T> signal, KParam sInfo,
                                  read_accessor<accType> impulse, int nBBS0,
                                  int nBBS1, const int FLEN, const int CONV_DIM,
                                  const bool EXPAND,
                                  sycl::local_accessor<T> localMem)
        : out_(out)
        , oInfo_(oInfo)
        , signal_(signal)
        , sInfo_(sInfo)
        , impulse_(impulse)
        , nBBS0_(nBBS0)
        , nBBS1_(nBBS1)
        , FLEN_(FLEN)
        , CONV_DIM_(CONV_DIM)
        , EXPAND_(EXPAND)
        , localMem_(localMem) {}
    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        const int radius  = FLEN_ - 1;
        const int padding = 2 * radius;
        const int s0      = sInfo_.strides[0];
        const int s1      = sInfo_.strides[1];
        const int d0      = sInfo_.dims[0];
        const int d1      = sInfo_.dims[1];
        const int shrdLen =
            g.get_local_range(0) + (CONV_DIM_ == 0 ? padding : 0);

        unsigned b2 = g.get_group_id(0) / nBBS0_;
        unsigned b3 = g.get_group_id(1) / nBBS1_;
        T *dst      = out_.get_pointer() +
                 (b2 * oInfo_.strides[2] + b3 * oInfo_.strides[3]);
        const T *src = signal_.get_pointer() +
                       (b2 * sInfo_.strides[2] + b3 * sInfo_.strides[3]) +
                       sInfo_.offset;

        int lx = it.get_local_id(0);
        int ly = it.get_local_id(1);
        int ox = g.get_local_range(0) * (g.get_group_id(0) - b2 * nBBS0_) + lx;
        int oy = g.get_local_range(1) * (g.get_group_id(1) - b3 * nBBS1_) + ly;
        int gx = ox;
        int gy = oy;

        // below if-else statement is based on MACRO value passed while kernel
        // compilation
        if (CONV_DIM_ == 0) {
            gx += (EXPAND_ ? 0 : FLEN_ >> 1);
            int endX = ((FLEN_ - 1) << 1) + g.get_local_range(0);
            for (int lx = it.get_local_id(0), glb_x = gx; lx < endX;
                 lx += g.get_local_range(0), glb_x += g.get_local_range(0)) {
                int i     = glb_x - radius;
                int j     = gy;
                bool is_i = i >= 0 && i < d0;
                bool is_j = j >= 0 && j < d1;
                localMem_[ly * shrdLen + lx] =
                    (is_i && is_j ? src[i * s0 + j * s1] : (T)(0));
            }

        } else if (CONV_DIM_ == 1) {
            gy += (EXPAND_ ? 0 : FLEN_ >> 1);
            int endY = ((FLEN_ - 1) << 1) + g.get_local_range(1);
            for (int ly = it.get_local_id(1), glb_y = gy; ly < endY;
                 ly += g.get_local_range(1), glb_y += g.get_local_range(1)) {
                int i     = gx;
                int j     = glb_y - radius;
                bool is_i = i >= 0 && i < d0;
                bool is_j = j >= 0 && j < d1;
                localMem_[ly * shrdLen + lx] =
                    (is_i && is_j ? src[i * s0 + j * s1] : (T)(0));
            }
        }
        it.barrier();

        if (ox < oInfo_.dims[0] && oy < oInfo_.dims[1]) {
            // below conditional statement is based on MACRO value passed while
            // kernel compilation
            int i         = (CONV_DIM_ == 0 ? lx : ly) + radius;
            accType accum = (accType)(0);
            for (int f = 0; f < FLEN_; ++f) {
                accType f_val = impulse_[f];
                // below conditional statement is based on MACRO value passed
                // while kernel compilation
                int s_idx = (CONV_DIM_ == 0 ? (ly * shrdLen + (i - f))
                                            : ((i - f) * shrdLen + lx));
                T s_val   = localMem_[s_idx];

                // binOp omitted from OpenCL implementation (see
                // convolve_separable.cl)
                accum = accum + (accType)s_val * (accType)f_val;
            }
            dst[oy * oInfo_.strides[1] + ox] = (T)accum;
        }
    }

   private:
    write_accessor<T> out_;
    KParam oInfo_;
    read_accessor<T> signal_;
    KParam sInfo_;
    read_accessor<accType> impulse_;
    int nBBS0_;
    int nBBS1_;
    const int FLEN_;
    const int CONV_DIM_;
    const bool EXPAND_;
    sycl::local_accessor<T> localMem_;
};

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

template<typename T, typename accType>
void convSep(Param<T> out, const Param<T> signal, const Param<accType> filter,
             const int conv_dim, const bool expand) {
    if (!(conv_dim == 0 || conv_dim == 1)) {
        AF_ERROR(
            "Separable convolution accepts only 0 or 1 as convolution "
            "dimension",
            AF_ERR_NOT_SUPPORTED);
    }
    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;

    const int fLen       = filter.info.dims[0] * filter.info.dims[1];
    const size_t C0_SIZE = (THREADS_X + 2 * (fLen - 1)) * THREADS_Y;
    const size_t C1_SIZE = (THREADS_Y + 2 * (fLen - 1)) * THREADS_X;
    size_t locSize       = (conv_dim == 0 ? C0_SIZE : C1_SIZE);

    auto local = sycl::range(THREADS_X, THREADS_Y);

    int blk_x = divup(out.info.dims[0], THREADS_X);
    int blk_y = divup(out.info.dims[1], THREADS_Y);

    auto global = sycl::range(blk_x * signal.info.dims[2] * THREADS_X,
                              blk_y * signal.info.dims[3] * THREADS_Y);

    sycl::buffer<accType> mBuff = {sycl::range(fLen * sizeof(accType))};
    memcpyBuffer(mBuff, *filter.data, fLen, 0);

    getQueue().submit([&](auto &h) {
        sycl::accessor d_signal{*signal.data, h, sycl::read_only};
        sycl::accessor d_out{*out.data, h, sycl::write_only, sycl::no_init};
        sycl::accessor d_mBuff{mBuff, h, sycl::read_only};
        sycl::local_accessor<T> localMem(locSize, h);
        h.parallel_for(sycl::nd_range{global, local},
                       convolveSeparableCreateKernel<T, accType>(
                           d_out, out.info, d_signal, signal.info, d_mBuff,
                           blk_x, blk_y, fLen, conv_dim, expand, localMem));
    });
}

#define INSTANTIATE(T, accT)                                          \
    template void convSep<T, accT>(Param<T>, const Param<T>,          \
                                   const Param<accT> filt, const int, \
                                   const bool);

INSTANTIATE(cdouble, cdouble)
INSTANTIATE(cfloat, cfloat)
INSTANTIATE(double, double)
INSTANTIATE(float, float)
INSTANTIATE(uint, float)
INSTANTIATE(int, float)
INSTANTIATE(uchar, float)
INSTANTIATE(char, float)
INSTANTIATE(ushort, float)
INSTANTIATE(short, float)
INSTANTIATE(uintl, float)
INSTANTIATE(intl, float)

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
