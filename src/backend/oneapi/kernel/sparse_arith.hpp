/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <Array.hpp>
#include <Param.hpp>
#include <common/Binary.hpp>
#include <common/complex.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <traits.hpp>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

constexpr unsigned TX      = 32;
constexpr unsigned TY      = 8;
constexpr unsigned THREADS = TX * TY;

template<typename T>
using global_atomic_ref =
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>;

template<typename T, af_op_t op>
class sparseArithCSRKernel {
   public:
    sparseArithCSRKernel(write_accessor<T> oPtr, const KParam out,
                         read_accessor<T> values, read_accessor<int> rowIdx,
                         read_accessor<int> colIdx, const int nNZ,
                         read_accessor<T> rPtr, const KParam rhs,
                         const int reverse)
        : oPtr_(oPtr)
        , out_(out)
        , values_(values)
        , rowIdx_(rowIdx)
        , colIdx_(colIdx)
        , nNZ_(nNZ)
        , rPtr_(rPtr)
        , rhs_(rhs)
        , reverse_(reverse) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        common::Binary<T, op> binOP;

        const int row =
            g.get_group_id(0) * g.get_local_range(1) + it.get_local_id(1);

        if (row < out_.dims[0]) {
            const int rowStartIdx = rowIdx_[row];
            const int rowEndIdx   = rowIdx_[row + 1];

            // Repeat loop until all values in the row are computed
            for (int idx = rowStartIdx + it.get_local_id(0); idx < rowEndIdx;
                 idx += g.get_local_range(0)) {
                const int col = colIdx_[idx];

                if (row >= out_.dims[0] || col >= out_.dims[1])
                    continue;  // Bad indices

                // Get Values
                const T val  = values_[idx];
                const T rval = rPtr_[col * rhs_.strides[1] + row];

                const int offset = col * out_.strides[1] + row;
                if (reverse_)
                    oPtr_[offset] = binOP(rval, val);
                else
                    oPtr_[offset] = binOP(val, rval);
            }
        }
    }

   private:
    write_accessor<T> oPtr_;
    const KParam out_;
    read_accessor<T> values_;
    read_accessor<int> rowIdx_;
    read_accessor<int> colIdx_;
    const int nNZ_;
    read_accessor<T> rPtr_;
    const KParam rhs_;
    const int reverse_;
};

template<typename T, af_op_t op>
void sparseArithOpCSR(Param<T> out, const Param<T> values,
                      const Param<int> rowIdx, const Param<int> colIdx,
                      const Param<T> rhs, const bool reverse) {
    auto local  = sycl::range(TX, TY);
    auto global = sycl::range(divup(out.info.dims[0], TY) * TX, TY);

    getQueue().submit([&](auto &h) {
        sycl::accessor d_out{*out.data, h, sycl::write_only};
        sycl::accessor d_values{*values.data, h, sycl::read_only};
        sycl::accessor d_rowIdx{*rowIdx.data, h, sycl::read_only};
        sycl::accessor d_colIdx{*colIdx.data, h, sycl::read_only};
        sycl::accessor d_rhs{*rhs.data, h, sycl::read_only};

        h.parallel_for(sycl::nd_range{global, local},
                       sparseArithCSRKernel<T, op>(
                           d_out, out.info, d_values, d_rowIdx, d_colIdx,
                           static_cast<int>(values.info.dims[0]), d_rhs,
                           rhs.info, static_cast<int>(reverse)));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
class sparseArithCOOKernel {
   public:
    sparseArithCOOKernel(write_accessor<T> oPtr, const KParam out,
                         read_accessor<T> values, read_accessor<int> rowIdx,
                         read_accessor<int> colIdx, const int nNZ,
                         read_accessor<T> rPtr, const KParam rhs,
                         const int reverse)
        : oPtr_(oPtr)
        , out_(out)
        , values_(values)
        , rowIdx_(rowIdx)
        , colIdx_(colIdx)
        , nNZ_(nNZ)
        , rPtr_(rPtr)
        , rhs_(rhs)
        , reverse_(reverse) {}

    void operator()(sycl::nd_item<1> it) const {
        common::Binary<T, op> binOP;

        const int idx = it.get_global_id(0);

        if (idx < nNZ_) {
            const int row = rowIdx_[idx];
            const int col = colIdx_[idx];

            if (row >= out_.dims[0] || col >= out_.dims[1])
                return;  // Bad indices

            // Get Values
            const T val  = values_[idx];
            const T rval = rPtr_[col * rhs_.strides[1] + row];

            const int offset = col * out_.strides[1] + row;
            if (reverse_)
                oPtr_[offset] = binOP(rval, val);
            else
                oPtr_[offset] = binOP(val, rval);
        }
    }

   private:
    write_accessor<T> oPtr_;
    const KParam out_;
    read_accessor<T> values_;
    read_accessor<int> rowIdx_;
    read_accessor<int> colIdx_;
    const int nNZ_;
    read_accessor<T> rPtr_;
    const KParam rhs_;
    const int reverse_;
};

template<typename T, af_op_t op>
void sparseArithOpCOO(Param<T> out, const Param<T> values,
                      const Param<int> rowIdx, const Param<int> colIdx,
                      const Param<T> rhs, const bool reverse) {
    auto local  = sycl::range(THREADS);
    auto global = sycl::range(divup(values.info.dims[0], THREADS) * THREADS);

    getQueue().submit([&](auto &h) {
        sycl::accessor d_out{*out.data, h, sycl::write_only};
        sycl::accessor d_values{*values.data, h, sycl::read_only};
        sycl::accessor d_rowIdx{*rowIdx.data, h, sycl::read_only};
        sycl::accessor d_colIdx{*colIdx.data, h, sycl::read_only};
        sycl::accessor d_rhs{*rhs.data, h, sycl::read_only};

        h.parallel_for(sycl::nd_range{global, local},
                       sparseArithCOOKernel<T, op>(
                           d_out, out.info, d_values, d_rowIdx, d_colIdx,
                           static_cast<int>(values.info.dims[0]), d_rhs,
                           rhs.info, static_cast<int>(reverse)));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
class sparseArithCSR2Kernel {
   public:
    sparseArithCSR2Kernel(sycl::accessor<T> values, read_accessor<int> rowIdx,
                          read_accessor<int> colIdx, const int nNZ,
                          read_accessor<T> rPtr, const KParam rhs,
                          const int reverse)
        : values_(values)
        , rowIdx_(rowIdx)
        , colIdx_(colIdx)
        , nNZ_(nNZ)
        , rPtr_(rPtr)
        , rhs_(rhs)
        , reverse_(reverse) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();
        common::Binary<T, op> binOP;

        const int row =
            g.get_group_id(0) * g.get_local_range(1) + it.get_local_id(1);

        if (row < rhs_.dims[0]) {
            const int rowStartIdx = rowIdx_[row];
            const int rowEndIdx   = rowIdx_[row + 1];

            // Repeat loop until all values in the row are computed
            for (int idx = rowStartIdx + it.get_local_id(0); idx < rowEndIdx;
                 idx += g.get_local_range(0)) {
                const int col = colIdx_[idx];

                if (row >= rhs_.dims[0] || col >= rhs_.dims[1])
                    continue;  // Bad indices

                // Get Values
                const T val  = values_[idx];
                const T rval = rPtr_[col * rhs_.strides[1] + row];

                if (reverse_)
                    values_[idx] = binOP(rval, val);
                else
                    values_[idx] = binOP(val, rval);
            }
        }
    }

   private:
    sycl::accessor<T> values_;
    read_accessor<int> rowIdx_;
    read_accessor<int> colIdx_;
    const int nNZ_;
    read_accessor<T> rPtr_;
    const KParam rhs_;
    const int reverse_;
};

template<typename T, af_op_t op>
void sparseArithOpCSR(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
                      const Param<T> rhs, const bool reverse) {
    auto local  = sycl::range(TX, TY);
    auto global = sycl::range(divup(values.info.dims[0], TY) * TX, TY);

    getQueue().submit([&](auto &h) {
        sycl::accessor d_values{*values.data, h, sycl::read_write};
        sycl::accessor d_rowIdx{*rowIdx.data, h, sycl::read_only};
        sycl::accessor d_colIdx{*colIdx.data, h, sycl::read_only};
        sycl::accessor d_rhs{*rhs.data, h, sycl::read_only};

        h.parallel_for(sycl::nd_range{global, local},
                       sparseArithCSR2Kernel<T, op>(
                           d_values, d_rowIdx, d_colIdx,
                           static_cast<int>(values.info.dims[0]), d_rhs,
                           rhs.info, static_cast<int>(reverse)));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
class sparseArithCOO2Kernel {
   public:
    sparseArithCOO2Kernel(sycl::accessor<T> values, read_accessor<int> rowIdx,
                          read_accessor<int> colIdx, const int nNZ,
                          read_accessor<T> rPtr, const KParam rhs,
                          const int reverse)
        : values_(values)
        , rowIdx_(rowIdx)
        , colIdx_(colIdx)
        , nNZ_(nNZ)
        , rPtr_(rPtr)
        , rhs_(rhs)
        , reverse_(reverse) {}

    void operator()(sycl::nd_item<1> it) const {
        common::Binary<T, op> binOP;

        const int idx = it.get_global_id(0);

        if (idx < nNZ_) {
            const int row = rowIdx_[idx];
            const int col = colIdx_[idx];

            if (row >= rhs_.dims[0] || col >= rhs_.dims[1])
                return;  // Bad indices

            // Get Values
            const T val  = values_[idx];
            const T rval = rPtr_[col * rhs_.strides[1] + row];

            if (reverse_)
                values_[idx] = binOP(rval, val);
            else
                values_[idx] = binOP(val, rval);
        }
    }

   private:
    sycl::accessor<T> values_;
    read_accessor<int> rowIdx_;
    read_accessor<int> colIdx_;
    const int nNZ_;
    read_accessor<T> rPtr_;
    const KParam rhs_;
    const int reverse_;
};

template<typename T, af_op_t op>
void sparseArithOpCOO(Param<T> values, Param<int> rowIdx, Param<int> colIdx,
                      const Param<T> rhs, const bool reverse) {
    auto local  = sycl::range(THREADS);
    auto global = sycl::range(divup(values.info.dims[0], THREADS) * THREADS);

    getQueue().submit([&](auto &h) {
        sycl::accessor d_values{*values.data, h, sycl::read_write};
        sycl::accessor d_rowIdx{*rowIdx.data, h, sycl::read_only};
        sycl::accessor d_colIdx{*colIdx.data, h, sycl::read_only};
        sycl::accessor d_rhs{*rhs.data, h, sycl::read_only};

        h.parallel_for(sycl::nd_range{global, local},
                       sparseArithCOO2Kernel<T, op>(
                           d_values, d_rowIdx, d_colIdx,
                           static_cast<int>(values.info.dims[0]), d_rhs,
                           rhs.info, static_cast<int>(reverse)));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

class csrCalcOutNNZKernel {
   public:
    csrCalcOutNNZKernel(write_accessor<unsigned> nnzc,
                        write_accessor<int> oRowIdx, unsigned M,
                        read_accessor<int> lRowIdx, read_accessor<int> lColIdx,
                        read_accessor<int> rRowIdx, read_accessor<int> rColIdx,
                        sycl::local_accessor<unsigned, 1> blkNNZ)
        : nnzc_(nnzc)
        , oRowIdx_(oRowIdx)
        , M_(M)
        , lRowIdx_(lRowIdx)
        , lColIdx_(lColIdx)
        , rRowIdx_(rRowIdx)
        , rColIdx_(rColIdx)
        , blkNNZ_(blkNNZ) {}

    void operator()(sycl::nd_item<1> it) const {
        sycl::group g = it.get_group();

        const uint row = it.get_global_id(0);
        const uint tid = it.get_local_id(0);

        const bool valid = row < M_;

        const uint lEnd = (valid ? lRowIdx_[row + 1] : 0);
        const uint rEnd = (valid ? rRowIdx_[row + 1] : 0);

        blkNNZ_[tid] = 0;
        it.barrier();

        uint l   = (valid ? lRowIdx_[row] : 0);
        uint r   = (valid ? rRowIdx_[row] : 0);
        uint nnz = 0;
        while (l < lEnd && r < rEnd) {
            uint lci = lColIdx_[l];
            uint rci = rColIdx_[r];
            l += (lci <= rci);
            r += (lci >= rci);
            nnz++;
        }
        nnz += (lEnd - l);
        nnz += (rEnd - r);

        blkNNZ_[tid] = nnz;
        it.barrier();

        if (valid) oRowIdx_[row + 1] = nnz;

        for (uint s = g.get_local_range(0) / 2; s > 0; s >>= 1) {
            if (tid < s) { blkNNZ_[tid] += blkNNZ_[tid + s]; }
            it.barrier();
        }

        if (tid == 0) {
            nnz = blkNNZ_[0];
            global_atomic_ref<uint>(nnzc_[0]) += nnz;
        }
    }

   private:
    write_accessor<unsigned> nnzc_;
    write_accessor<int> oRowIdx_;
    unsigned M_;
    read_accessor<int> lRowIdx_;
    read_accessor<int> lColIdx_;
    read_accessor<int> rRowIdx_;
    read_accessor<int> rColIdx_;
    sycl::local_accessor<unsigned, 1> blkNNZ_;
};

static void csrCalcOutNNZ(Param<int> outRowIdx, unsigned &nnzC, const uint M,
                          const uint N, uint nnzA, const Param<int> lrowIdx,
                          const Param<int> lcolIdx, uint nnzB,
                          const Param<int> rrowIdx, const Param<int> rcolIdx) {
    UNUSED(N);
    UNUSED(nnzA);
    UNUSED(nnzB);

    auto local  = sycl::range(256);
    auto global = sycl::range(divup(M, local[0]) * local[0]);

    Array<unsigned> out = createValueArray<unsigned>(1, 0);
    auto out_get = out.get();

    getQueue().submit([&](auto &h) {
        sycl::accessor d_out{*out_get, h, sycl::write_only};
        sycl::accessor d_outRowIdx{*outRowIdx.data, h, sycl::write_only};
        sycl::accessor d_lRowIdx{*lrowIdx.data, h, sycl::read_only};
        sycl::accessor d_lColIdx{*lcolIdx.data, h, sycl::read_only};
        sycl::accessor d_rRowIdx{*rrowIdx.data, h, sycl::read_only};
        sycl::accessor d_rColIdx{*rcolIdx.data, h, sycl::read_only};

        auto blkNNZ = sycl::local_accessor<unsigned, 1>(local[0], h);
        h.parallel_for(
            sycl::nd_range{global, local},
            csrCalcOutNNZKernel(d_out, d_outRowIdx, M, d_lRowIdx, d_lColIdx,
                                d_rRowIdx, d_rColIdx, blkNNZ));
    });

    {
        sycl::host_accessor nnz_acc{*out.get(), sycl::read_only};
        nnzC = nnz_acc[0];
    }

    ONEAPI_DEBUG_FINISH(getQueue());
}

template<typename T, af_op_t op>
class ssarithCSRKernel {
   public:
    ssarithCSRKernel(write_accessor<T> oVals, write_accessor<int> oColIdx,
                     read_accessor<int> oRowIdx, unsigned M, unsigned N,
                     unsigned nnza, read_accessor<T> lVals,
                     read_accessor<int> lRowIdx, read_accessor<int> lColIdx,
                     unsigned nnzb, read_accessor<T> rVals,
                     read_accessor<int> rRowIdx, read_accessor<int> rColIdx)
        : oVals_(oVals)
        , oColIdx_(oColIdx)
        , oRowIdx_(oRowIdx)
        , M_(M)
        , N_(N)
        , nnza_(nnza)
        , lVals_(lVals)
        , lRowIdx_(lRowIdx)
        , lColIdx_(lColIdx)
        , nnzb_(nnzb)
        , rVals_(rVals)
        , rRowIdx_(rRowIdx)
        , rColIdx_(rColIdx) {}

    void operator()(sycl::nd_item<1> it) const {
        common::Binary<T, op> binOP;

        const uint row = it.get_global_id(0);

        const bool valid  = row < M_;
        const uint lEnd   = (valid ? lRowIdx_[row + 1] : 0);
        const uint rEnd   = (valid ? rRowIdx_[row + 1] : 0);
        const uint offset = (valid ? oRowIdx_[row] : 0);

        T *ovPtr   = oVals_.get_pointer() + offset;
        int *ocPtr = oColIdx_.get_pointer() + offset;

        uint l = (valid ? lRowIdx_[row] : 0);
        uint r = (valid ? rRowIdx_[row] : 0);

        uint nnz = 0;
        while (l < lEnd && r < rEnd) {
            uint lci = lColIdx_[l];
            uint rci = rColIdx_[r];

            T lhs = (lci <= rci ? lVals_[l] : common::Binary<T, op>::init());
            T rhs = (lci >= rci ? rVals_[r] : common::Binary<T, op>::init());

            ovPtr[nnz] = binOP(lhs, rhs);
            ocPtr[nnz] = (lci <= rci) ? lci : rci;

            l += (lci <= rci);
            r += (lci >= rci);
            nnz++;
        }
        while (l < lEnd) {
            ovPtr[nnz] = binOP(lVals_[l], common::Binary<T, op>::init());
            ocPtr[nnz] = lColIdx_[l];
            l++;
            nnz++;
        }
        while (r < rEnd) {
            ovPtr[nnz] = binOP(common::Binary<T, op>::init(), rVals_[r]);
            ocPtr[nnz] = rColIdx_[r];
            r++;
            nnz++;
        }
    }

   private:
    write_accessor<T> oVals_;
    write_accessor<int> oColIdx_;
    read_accessor<int> oRowIdx_;
    unsigned M_, N_;
    unsigned nnza_;
    read_accessor<T> lVals_;
    read_accessor<int> lRowIdx_;
    read_accessor<int> lColIdx_;
    unsigned nnzb_;
    read_accessor<T> rVals_;
    read_accessor<int> rRowIdx_;
    read_accessor<int> rColIdx_;
};

template<typename T, af_op_t op>
void ssArithCSR(Param<T> oVals, Param<int> oColIdx, const Param<int> oRowIdx,
                const uint M, const uint N, unsigned nnzA, const Param<T> lVals,
                const Param<int> lRowIdx, const Param<int> lColIdx,
                unsigned nnzB, const Param<T> rVals, const Param<int> rRowIdx,
                const Param<int> rColIdx) {
    auto local  = sycl::range(256);
    auto global = sycl::range(divup(M, local[0]) * local[0]);

    getQueue().submit([&](auto &h) {
        sycl::accessor d_oVals{*oVals.data, h, sycl::write_only};
        sycl::accessor d_oColIdx{*oColIdx.data, h, sycl::write_only};
        sycl::accessor d_oRowIdx{*oRowIdx.data, h, sycl::read_only};

        sycl::accessor d_lVals{*lVals.data, h, sycl::read_only};
        sycl::accessor d_lRowIdx{*lRowIdx.data, h, sycl::read_only};
        sycl::accessor d_lColIdx{*lColIdx.data, h, sycl::read_only};

        sycl::accessor d_rVals{*rVals.data, h, sycl::read_only};
        sycl::accessor d_rRowIdx{*rRowIdx.data, h, sycl::read_only};
        sycl::accessor d_rColIdx{*rColIdx.data, h, sycl::read_only};

        h.parallel_for(
            sycl::nd_range{global, local},
            ssarithCSRKernel<T, op>(d_oVals, d_oColIdx, d_oRowIdx, M, N, nnzA,
                                    d_lVals, d_lRowIdx, d_lColIdx, nnzB,
                                    d_rVals, d_rRowIdx, d_rColIdx));
    });
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
