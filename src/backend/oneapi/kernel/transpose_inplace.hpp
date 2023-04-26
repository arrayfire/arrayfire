/*******************************************************
 * Copyright (c) 2022, ArrayFire
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
#include <err_oneapi.hpp>
#include <traits.hpp>

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
static T getConjugate(const T &in) {
    // For non-complex types return same
    return in;
}

template<>
cfloat getConjugate(const cfloat &in) {
    return std::conj(in);
}

template<>
cdouble getConjugate(const cdouble &in) {
    return std::conj(in);
}

#define doOp(v) (conjugate_ ? getConjugate((v)) : (v))

constexpr dim_t TILE_DIM  = 16;
constexpr dim_t THREADS_X = TILE_DIM;
constexpr dim_t THREADS_Y = 256 / TILE_DIM;

template<typename T>
class transposeInPlaceKernel {
   public:
    transposeInPlaceKernel(const sycl::accessor<T> iData, const KParam in,
                           const int blocksPerMatX, const int blocksPerMatY,
                           const bool conjugate, const bool IS32MULTIPLE,
                           sycl::local_accessor<T, 1> shrdMem_s,
                           sycl::local_accessor<T, 1> shrdMem_d)
        : iData_(iData)
        , in_(in)
        , blocksPerMatX_(blocksPerMatX)
        , blocksPerMatY_(blocksPerMatY)
        , conjugate_(conjugate)
        , IS32MULTIPLE_(IS32MULTIPLE)
        , shrdMem_s_(shrdMem_s)
        , shrdMem_d_(shrdMem_d) {}
    void operator()(sycl::nd_item<2> it) const {
        const int shrdStride = TILE_DIM + 1;

        // create variables to hold output dimensions
        const int iDim0 = in_.dims[0];
        const int iDim1 = in_.dims[1];

        // calculate strides
        const int iStride1 = in_.strides[1];

        const int lx = it.get_local_id(0);
        const int ly = it.get_local_id(1);

        // batch based block Id
        sycl::group g        = it.get_group();
        const int batchId_x  = g.get_group_id(0) / blocksPerMatX_;
        const int blockIdx_x = (g.get_group_id(0) - batchId_x * blocksPerMatX_);

        const int batchId_y  = g.get_group_id(1) / blocksPerMatY_;
        const int blockIdx_y = (g.get_group_id(1) - batchId_y * blocksPerMatY_);

        const int x0 = TILE_DIM * blockIdx_x;
        const int y0 = TILE_DIM * blockIdx_y;

        T *iDataPtr = iData_.get_pointer();
        iDataPtr += batchId_x * in_.strides[2] + batchId_y * in_.strides[3] +
                    in_.offset;

        if (blockIdx_y > blockIdx_x) {
            // calculate global indices
            int gx = lx + x0;
            int gy = ly + y0;
            int dx = lx + y0;
            int dy = ly + x0;

            // Copy to shared memory
            for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
                int gy_ = gy + repeat;
                if (IS32MULTIPLE_ || (gx < iDim0 && gy_ < iDim1))
                    shrdMem_s_[(ly + repeat) * shrdStride + lx] =
                        iDataPtr[gy_ * iStride1 + gx];

                int dy_ = dy + repeat;
                if (IS32MULTIPLE_ || (dx < iDim0 && dy_ < iDim1))
                    shrdMem_d_[(ly + repeat) * shrdStride + lx] =
                        iDataPtr[dy_ * iStride1 + dx];
            }

            it.barrier();

            // Copy from shared memory to global memory
            for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
                int dy_ = dy + repeat;
                if (IS32MULTIPLE_ || (dx < iDim0 && dy_ < iDim1))
                    iDataPtr[dy_ * iStride1 + dx] =
                        doOp(shrdMem_s_[(ly + repeat) + (shrdStride * lx)]);

                int gy_ = gy + repeat;
                if (IS32MULTIPLE_ || (gx < iDim0 && gy_ < iDim1))
                    iDataPtr[gy_ * iStride1 + gx] =
                        doOp(shrdMem_d_[(ly + repeat) + (shrdStride * lx)]);
            }

        } else if (blockIdx_y == blockIdx_x) {
            // calculate global indices
            int gx = lx + x0;
            int gy = ly + y0;

            // Copy to shared memory
            for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
                int gy_ = gy + repeat;
                if (IS32MULTIPLE_ || (gx < iDim0 && gy_ < iDim1))
                    shrdMem_s_[(ly + repeat) * shrdStride + lx] =
                        iDataPtr[gy_ * iStride1 + gx];
            }

            it.barrier();

            // Copy from shared memory to global memory
            for (int repeat = 0; repeat < TILE_DIM; repeat += THREADS_Y) {
                int gy_ = gy + repeat;
                if (IS32MULTIPLE_ || (gx < iDim0 && gy_ < iDim1))
                    iDataPtr[gy_ * iStride1 + gx] =
                        doOp(shrdMem_s_[(ly + repeat) + (shrdStride * lx)]);
            }
        }
    }

   private:
    sycl::accessor<T> iData_;
    KParam in_;
    int blocksPerMatX_;
    int blocksPerMatY_;
    bool conjugate_;
    bool IS32MULTIPLE_;
    sycl::local_accessor<T, 1> shrdMem_s_;
    sycl::local_accessor<T, 1> shrdMem_d_;
};

template<typename T>
void transpose_inplace(Param<T> in, const bool conjugate,
                       const bool IS32MULTIPLE) {
    auto local = sycl::range{THREADS_X, THREADS_Y};

    int blk_x = divup(in.info.dims[0], TILE_DIM);
    int blk_y = divup(in.info.dims[1], TILE_DIM);

    auto global = sycl::range{blk_x * local[0] * in.info.dims[2],
                              blk_y * local[1] * in.info.dims[3]};

    getQueue().submit([&](sycl::handler &h) {
        auto r = in.data->get_access(h);
        auto shrdMem_s =
            sycl::local_accessor<T, 1>(TILE_DIM * (TILE_DIM + 1), h);
        auto shrdMem_d =
            sycl::local_accessor<T, 1>(TILE_DIM * (TILE_DIM + 1), h);

        h.parallel_for(
            sycl::nd_range{global, local},
            transposeInPlaceKernel<T>(r, in.info, blk_x, blk_y, conjugate,
                                      IS32MULTIPLE, shrdMem_s, shrdMem_d));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
