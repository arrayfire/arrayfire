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
#include <backend.hpp>
#include <common/Binary.hpp>
#include <common/Transform.hpp>
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <kernel/reduce_config.hpp>
#include <math.hpp>
#include <memory.hpp>
#include <type_traits>

using std::unique_ptr;

namespace arrayfire {
namespace oneapi {
namespace kernel {

// Reduces keys across block boundaries
template<typename Tk, typename To, af_op_t op>
class finalBoundaryReduceKernel {
   public:
    finalBoundaryReduceKernel(write_accessor<int> reduced_block_sizes,
                              read_accessor<Tk> iKeys, KParam iKInfo,
                              sycl::accessor<To> oVals, KParam oVInfo,
                              const int n)
        : reduced_block_sizes_(reduced_block_sizes)
        , iKeys_(iKeys)
        , iKInfo_(iKInfo)
        , oVals_(oVals)
        , oVInfo_(oVInfo)
        , n_(n) {}

    void operator()(sycl::nd_item<1> it) const {
        sycl::group g  = it.get_group();
        const uint lid = it.get_local_id(0);
        const uint gid = it.get_global_id(0);
        const uint bid = g.get_group_id(0);

        common::Binary<compute_t<To>, op> binOp;
        if (gid == ((bid + 1) * it.get_local_range(0)) - 1 &&
            bid < g.get_group_range(0) - 1) {
            Tk k0 = iKeys_[gid];
            Tk k1 = iKeys_[gid + 1];

            if (k0 == k1) {
                compute_t<To> v0          = compute_t<To>(oVals_[gid]);
                compute_t<To> v1          = compute_t<To>(oVals_[gid + 1]);
                oVals_[gid + 1]           = binOp(v0, v1);
                reduced_block_sizes_[bid] = it.get_local_range(0) - 1;
            } else {
                reduced_block_sizes_[bid] = it.get_local_range(0);
            }
        }

        // if last block, set block size to difference between n and block
        // boundary
        if (lid == 0 && bid == g.get_group_range(0) - 1) {
            reduced_block_sizes_[bid] = n_ - (bid * it.get_local_range(0));
        }
    }

   protected:
    write_accessor<int> reduced_block_sizes_;
    read_accessor<Tk> iKeys_;
    KParam iKInfo_;
    sycl::accessor<To> oVals_;
    KParam oVInfo_;
    int n_;
};

template<typename Tk, typename To, af_op_t op>
class finalBoundaryReduceDimKernel {
   public:
    finalBoundaryReduceDimKernel(write_accessor<int> reduced_block_sizes,
                                 read_accessor<Tk> iKeys, KParam iKInfo,
                                 sycl::accessor<To> oVals, KParam oVInfo,
                                 const int n, const int nGroupsZ)
        : reduced_block_sizes_(reduced_block_sizes)
        , iKeys_(iKeys)
        , iKInfo_(iKInfo)
        , oVals_(oVals)
        , oVInfo_(oVInfo)
        , n_(n)
        , nGroupsZ_(nGroupsZ) {}

    void operator()(sycl::nd_item<3> it) const {
        sycl::group g  = it.get_group();
        const uint lid = it.get_local_id(0);
        const uint gid = it.get_global_id(0);
        const uint bid = g.get_group_id(0);

        common::Binary<compute_t<To>, op> binOp;
        if (gid == ((bid + 1) * it.get_local_range(0)) - 1 &&
            bid < g.get_group_range(0) - 1) {
            Tk k0 = iKeys_[gid];
            Tk k1 = iKeys_[gid + 1];

            if (k0 == k1) {
                compute_t<To> v0          = compute_t<To>(oVals_[gid]);
                compute_t<To> v1          = compute_t<To>(oVals_[gid + 1]);
                oVals_[gid + 1]           = binOp(v0, v1);
                reduced_block_sizes_[bid] = it.get_local_range(0) - 1;
            } else {
                reduced_block_sizes_[bid] = it.get_local_range(0);
            }
        }

        // if last block, set block size to difference between n and block
        // boundary
        if (lid == 0 && bid == g.get_group_range(0) - 1) {
            reduced_block_sizes_[bid] = n_ - (bid * it.get_local_range(0));
        }
    }

   protected:
    write_accessor<int> reduced_block_sizes_;
    read_accessor<Tk> iKeys_;
    KParam iKInfo_;
    sycl::accessor<To> oVals_;
    KParam oVInfo_;
    int n_;
    int nGroupsZ_;
};

template<typename T>
using global_atomic_ref =
    sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::system,
                     sycl::access::address_space::global_space>;

// Tests if data needs further reduction, including across block boundaries
template<typename Tk>
class testNeedsReductionKernel {
   public:
    testNeedsReductionKernel(sycl::accessor<int> needs_another_reduction,
                             sycl::accessor<int> needs_block_boundary_reduced,
                             read_accessor<Tk> iKeys, KParam iKInfo,
                             const int n, const int DIMX,
                             sycl::local_accessor<Tk> l_keys)
        : needs_another_reduction_(needs_another_reduction)
        , needs_block_boundary_reduced_(needs_block_boundary_reduced)
        , iKeys_(iKeys)
        , iKInfo_(iKInfo)
        , n_(n)
        , DIMX_(DIMX)
        , l_keys_(l_keys) {}

    void operator()(sycl::nd_item<1> it) const {
        sycl::group g  = it.get_group();
        const uint lid = it.get_local_id(0);
        const uint gid = it.get_global_id(0);
        const uint bid = g.get_group_id(0);

        Tk k = scalar<Tk>(0);
        if (gid < n_) { k = iKeys_[gid]; }

        l_keys_[lid] = k;
        it.barrier();

        int update_key =
            (lid < DIMX_ - 2) && (k == l_keys_[lid + 1]) && (gid < (n_ - 1));

        if (update_key) {
            global_atomic_ref<int>(needs_another_reduction_[0]) |= update_key;
        }

        it.barrier();

        // last thread in each block checks if any inter-block keys need further
        // reduction
        if (gid == ((bid + 1) * DIMX_) - 1 &&
            bid < (g.get_group_range(0) - 1)) {
            int k0 = iKeys_[gid];
            int k1 = iKeys_[gid + 1];
            if (k0 == k1) {
                global_atomic_ref<int>(needs_block_boundary_reduced_[0]) |= 1;
            }
        }
    }

   protected:
    sycl::accessor<int> needs_another_reduction_;
    sycl::accessor<int> needs_block_boundary_reduced_;
    read_accessor<Tk> iKeys_;
    KParam iKInfo_;
    int n_;
    int DIMX_;
    sycl::local_accessor<Tk> l_keys_;
};

// Compacts "incomplete" block-sized chunks of data in global memory
template<typename Tk, typename To>
class compactKernel {
   public:
    compactKernel(read_accessor<int> reduced_block_sizes,
                  write_accessor<Tk> oKeys, KParam oKInfo,
                  write_accessor<To> oVals, KParam oVInfo,
                  read_accessor<Tk> iKeys, KParam iKInfo,
                  read_accessor<To> iVals, KParam iVInfo, int nGroupsZ)
        : reduced_block_sizes_(reduced_block_sizes)
        , oKeys_(oKeys)
        , oKInfo_(oKInfo)
        , oVals_(oVals)
        , oVInfo_(oVInfo)
        , iKeys_(iKeys)
        , iKInfo_(iKInfo)
        , iVals_(iVals)
        , iVInfo_(iVInfo)
        , nGroupsZ_(nGroupsZ) {}

    void operator()(sycl::nd_item<3> it) const {
        sycl::group g  = it.get_group();
        const uint lid = it.get_local_id(0);
        const uint bid = g.get_group_id(0);
        const uint gid = it.get_global_id(0);

        const int bidy = g.get_group_id(1);
        const int bidz = g.get_group_id(2) % nGroupsZ_;
        const int bidw = g.get_group_id(2) / nGroupsZ_;

        const int bOffset = bidw * oVInfo_.strides[3] +
                            bidz * oVInfo_.strides[2] +
                            bidy * oVInfo_.strides[1];

        // reduced_block_sizes should have inclusive sum of block sizes
        int nwrite =
            (bid == 0)
                ? reduced_block_sizes_[0]
                : (reduced_block_sizes_[bid] - reduced_block_sizes_[bid - 1]);
        int writeloc = (bid == 0) ? 0 : reduced_block_sizes_[bid - 1];

        Tk k = iKeys_[gid];
        To v = iVals_[bOffset + gid];

        if (lid < nwrite) {
            oKeys_[writeloc + lid]           = k;
            oVals_[bOffset + writeloc + lid] = v;
        }
    }

   protected:
    read_accessor<int> reduced_block_sizes_;
    write_accessor<Tk> oKeys_;
    KParam oKInfo_;
    write_accessor<To> oVals_;
    KParam oVInfo_;
    read_accessor<Tk> iKeys_;
    KParam iKInfo_;
    read_accessor<To> iVals_;
    KParam iVInfo_;
    int nGroupsZ_;
};

// Compacts "incomplete" block-sized chunks of data in global memory
template<typename Tk, typename To>
class compactDimKernel {
   public:
    compactDimKernel(read_accessor<int> reduced_block_sizes,
                     write_accessor<Tk> oKeys, KParam oKInfo,
                     write_accessor<To> oVals, KParam oVInfo,
                     read_accessor<Tk> iKeys, KParam iKInfo,
                     read_accessor<To> iVals, KParam iVInfo, int nGroupsZ,
                     int DIM)
        : reduced_block_sizes_(reduced_block_sizes)
        , oKeys_(oKeys)
        , oKInfo_(oKInfo)
        , oVals_(oVals)
        , oVInfo_(oVInfo)
        , iKeys_(iKeys)
        , iKInfo_(iKInfo)
        , iVals_(iVals)
        , iVInfo_(iVInfo)
        , nGroupsZ_(nGroupsZ)
        , DIM_(DIM) {}

    void operator()(sycl::nd_item<3> it) const {
        sycl::group g = it.get_group();

        const uint lid  = it.get_local_id(0);
        const uint gidx = it.get_global_id(0);
        const uint bid  = g.get_group_id(0);

        const int bidy = g.get_group_id(1);
        const int bidz = g.get_group_id(2) % nGroupsZ_;
        const int bidw = g.get_group_id(2) / nGroupsZ_;

        int dims_ordering[4];
        dims_ordering[0] = DIM_;
        int d            = 1;
        for (int i = 0; i < 4; ++i) {
            if (i != DIM_) dims_ordering[d++] = i;
        }

        Tk k;
        To v;

        // reduced_block_sizes should have inclusive sum of block sizes
        int nwrite =
            (bid == 0)
                ? reduced_block_sizes_[0]
                : (reduced_block_sizes_[bid] - reduced_block_sizes_[bid - 1]);
        int writeloc = (bid == 0) ? 0 : reduced_block_sizes_[bid - 1];

        const int tid = bidw * iVInfo_.strides[dims_ordering[3]] +
                        bidz * iVInfo_.strides[dims_ordering[2]] +
                        bidy * iVInfo_.strides[dims_ordering[1]] +
                        gidx * iVInfo_.strides[DIM_];
        k = iKeys_[gidx];
        v = iVals_[tid];

        if (lid < nwrite) {
            oKeys_[writeloc + lid] = k;
            const int bOffset      = bidw * oVInfo_.strides[dims_ordering[3]] +
                                bidz * oVInfo_.strides[dims_ordering[2]] +
                                bidy * oVInfo_.strides[dims_ordering[1]];
            oVals_[bOffset + (writeloc + lid) * oVInfo_.strides[DIM_]] = v;
        }
    }

   protected:
    read_accessor<int> reduced_block_sizes_;
    write_accessor<Tk> oKeys_;
    KParam oKInfo_;
    write_accessor<To> oVals_;
    KParam oVInfo_;
    read_accessor<Tk> iKeys_;
    KParam iKInfo_;
    read_accessor<To> iVals_;
    KParam iVInfo_;
    int nGroupsZ_;
    int DIM_;
};

// Reduces each block by key
template<typename Ti, typename Tk, typename To, af_op_t op>
class reduceBlocksByKeyKernel {
   public:
    reduceBlocksByKeyKernel(sycl::accessor<int> reduced_block_sizes,
                            write_accessor<Tk> oKeys, KParam oKInfo,
                            write_accessor<To> oVals, KParam oVInfo,
                            read_accessor<Tk> iKeys, KParam iKInfo,
                            read_accessor<Ti> iVals, KParam iVInfo,
                            int change_nan, To nanval, int n, int nGroupsZ,
                            int DIMX, sycl::local_accessor<Tk> l_keys,
                            sycl::local_accessor<compute_t<To>> l_vals,
                            sycl::local_accessor<Tk> l_reduced_keys,
                            sycl::local_accessor<compute_t<To>> l_reduced_vals,
                            sycl::local_accessor<int> l_unique_ids,
                            sycl::local_accessor<int> l_wg_temp,
                            sycl::local_accessor<int> l_unique_flags,
                            sycl::local_accessor<int> l_reduced_block_size)
        : reduced_block_sizes_(reduced_block_sizes)
        , oKeys_(oKeys)
        , oKInfo_(oKInfo)
        , oVals_(oVals)
        , oVInfo_(oVInfo)
        , iKeys_(iKeys)
        , iKInfo_(iKInfo)
        , iVals_(iVals)
        , iVInfo_(iVInfo)
        , change_nan_(change_nan)
        , nanval_(nanval)
        , n_(n)
        , nGroupsZ_(nGroupsZ)
        , DIMX_(DIMX)
        , l_keys_(l_keys)
        , l_vals_(l_vals)
        , l_reduced_keys_(l_reduced_keys)
        , l_reduced_vals_(l_reduced_vals)
        , l_unique_ids_(l_unique_ids)
        , l_wg_temp_(l_wg_temp)
        , l_unique_flags_(l_unique_flags)
        , l_reduced_block_size_(l_reduced_block_size) {}

    void operator()(sycl::nd_item<3> it) const {
        sycl::group g  = it.get_group();
        const uint lid = it.get_local_id(0);
        const uint gid = it.get_global_id(0);

        const int bidy = g.get_group_id(1);
        const int bidz = g.get_group_id(2) % nGroupsZ_;
        const int bidw = g.get_group_id(2) / nGroupsZ_;

        const compute_t<To> init_val =
            common::Binary<compute_t<To>, op>::init();
        common::Binary<compute_t<To>, op> binOp;
        common::Transform<Ti, compute_t<To>, op> transform;

        if (lid == 0) { l_reduced_block_size_[0] = 0; }

        // load keys and values to threads
        Tk k            = scalar<Tk>(0);
        compute_t<To> v = init_val;
        if (gid < n_) {
            k                 = iKeys_[gid];
            const int bOffset = bidw * iVInfo_.strides[3] +
                                bidz * iVInfo_.strides[2] +
                                bidy * iVInfo_.strides[1];
            v = transform(iVals_[bOffset + gid]);
            if (change_nan_) v = IS_NAN(v) ? nanval_ : v;
        }

        l_keys_[lid] = k;
        l_vals_[lid] = v;

        l_reduced_keys_[lid] = k;
        it.barrier();

        // mark threads containing unique keys
        int eq_check    = (lid > 0) ? (k != l_reduced_keys_[lid - 1]) : 0;
        int unique_flag = (eq_check || (lid == 0)) && (gid < n_);

        l_unique_flags_[lid] = unique_flag;
        int unique_id =
            work_group_scan_inclusive_add(it, l_wg_temp_, l_unique_flags_);

        l_unique_ids_[lid] = unique_id;

        if (lid == DIMX_ - 1) l_reduced_block_size_[0] = unique_id;

        for (int off = 1; off < DIMX_; off *= 2) {
            it.barrier();
            int test_unique_id =
                (lid + off < DIMX_) ? l_unique_ids_[lid + off] : ~unique_id;
            eq_check = (unique_id == test_unique_id);
            int update_key =
                eq_check && (lid < (DIMX_ - off)) &&
                ((gid + off) <
                 n_);  // checks if this thread should perform a reduction
            compute_t<To> uval = (update_key) ? l_vals_[lid + off] : init_val;
            it.barrier();
            l_vals_[lid] =
                binOp(l_vals_[lid], uval);  // update if thread requires it
        }

        if (unique_flag) {
            l_reduced_keys_[unique_id - 1] = k;
            l_reduced_vals_[unique_id - 1] = l_vals_[lid];
        }
        it.barrier();

        const int bid = g.get_group_id(0);
        if (lid < l_reduced_block_size_[0]) {
            const int bOffset = bidw * oVInfo_.strides[3] +
                                bidz * oVInfo_.strides[2] +
                                bidy * oVInfo_.strides[1];
            oKeys_[bid * DIMX_ + lid]               = l_reduced_keys_[lid];
            oVals_[bOffset + ((bid * DIMX_) + lid)] = l_reduced_vals_[lid];
        }

        reduced_block_sizes_[bid] = l_reduced_block_size_[0];
    }

    int work_group_scan_inclusive_add(sycl::nd_item<3> it,
                                      sycl::local_accessor<int> wg_temp,
                                      sycl::local_accessor<int> arr) const {
        const uint lid = it.get_local_id(0);
        int *active_buf;

        int val    = arr[lid];
        active_buf = arr.get_pointer();

        bool swap_buffer = false;
        for (int off = 1; off <= DIMX_; off *= 2) {
            it.barrier();
            if (lid >= off) { val = val + active_buf[lid - off]; }
            swap_buffer = !swap_buffer;
            active_buf =
                swap_buffer ? wg_temp.get_pointer() : arr.get_pointer();
            active_buf[lid] = val;
        }

        int res = active_buf[lid];
        return res;
    }

   protected:
    sycl::accessor<int> reduced_block_sizes_;
    write_accessor<Tk> oKeys_;
    KParam oKInfo_;
    write_accessor<To> oVals_;
    KParam oVInfo_;
    read_accessor<Tk> iKeys_;
    KParam iKInfo_;
    read_accessor<Ti> iVals_;
    KParam iVInfo_;
    int change_nan_;
    To nanval_;
    int n_;
    int nGroupsZ_;
    int DIMX_;
    sycl::local_accessor<Tk> l_keys_;
    sycl::local_accessor<compute_t<To>> l_vals_;
    sycl::local_accessor<Tk> l_reduced_keys_;
    sycl::local_accessor<compute_t<To>> l_reduced_vals_;
    sycl::local_accessor<int> l_unique_ids_;
    sycl::local_accessor<int> l_wg_temp_;
    sycl::local_accessor<int> l_unique_flags_;
    sycl::local_accessor<int> l_reduced_block_size_;
};

// Reduces each block by key
template<typename Ti, typename Tk, typename To, af_op_t op>
class reduceBlocksByKeyDimKernel {
   public:
    reduceBlocksByKeyDimKernel(
        sycl::accessor<int> reduced_block_sizes, write_accessor<Tk> oKeys,
        KParam oKInfo, write_accessor<To> oVals, KParam oVInfo,
        read_accessor<Tk> iKeys, KParam iKInfo, read_accessor<Ti> iVals,
        KParam iVInfo, int change_nan, To nanval, int n, int nGroupsZ, int DIMX,
        int DIM, sycl::local_accessor<Tk> l_keys,
        sycl::local_accessor<compute_t<To>> l_vals,
        sycl::local_accessor<Tk> l_reduced_keys,
        sycl::local_accessor<compute_t<To>> l_reduced_vals,
        sycl::local_accessor<int> l_unique_ids,
        sycl::local_accessor<int> l_wg_temp,
        sycl::local_accessor<int> l_unique_flags,
        sycl::local_accessor<int> l_reduced_block_size)
        : reduced_block_sizes_(reduced_block_sizes)
        , oKeys_(oKeys)
        , oKInfo_(oKInfo)
        , oVals_(oVals)
        , oVInfo_(oVInfo)
        , iKeys_(iKeys)
        , iKInfo_(iKInfo)
        , iVals_(iVals)
        , iVInfo_(iVInfo)
        , change_nan_(change_nan)
        , nanval_(nanval)
        , n_(n)
        , nGroupsZ_(nGroupsZ)
        , DIMX_(DIMX)
        , DIM_(DIM)
        , l_keys_(l_keys)
        , l_vals_(l_vals)
        , l_reduced_keys_(l_reduced_keys)
        , l_reduced_vals_(l_reduced_vals)
        , l_unique_ids_(l_unique_ids)
        , l_wg_temp_(l_wg_temp)
        , l_unique_flags_(l_unique_flags)
        , l_reduced_block_size_(l_reduced_block_size) {}

    void operator()(sycl::nd_item<3> it) const {
        sycl::group g  = it.get_group();
        const uint lid = it.get_local_id(0);
        const uint gid = it.get_global_id(0);

        const int bidy = g.get_group_id(1);
        const int bidz = g.get_group_id(2) % nGroupsZ_;
        const int bidw = g.get_group_id(2) / nGroupsZ_;

        const compute_t<To> init_val =
            common::Binary<compute_t<To>, op>::init();
        common::Binary<compute_t<To>, op> binOp;
        common::Transform<Ti, compute_t<To>, op> transform;

        if (lid == 0) { l_reduced_block_size_[0] = 0; }

        int dims_ordering[4];
        dims_ordering[0] = DIM_;
        int d            = 1;
        for (int i = 0; i < 4; ++i) {
            if (i != DIM_) dims_ordering[d++] = i;
        }
        it.barrier();

        // load keys and values to threads
        Tk k            = scalar<Tk>(0);
        compute_t<To> v = init_val;
        if (gid < n_) {
            k                 = iKeys_[gid];
            const int bOffset = bidw * iVInfo_.strides[dims_ordering[3]] +
                                bidz * iVInfo_.strides[dims_ordering[2]] +
                                bidy * iVInfo_.strides[dims_ordering[1]];
            v = transform(iVals_[bOffset + gid * iVInfo_.strides[DIM_]]);
            if (change_nan_) v = IS_NAN(v) ? nanval_ : v;
        }

        l_keys_[lid] = k;
        l_vals_[lid] = v;

        l_reduced_keys_[lid] = k;
        it.barrier();

        // mark threads containing unique keys
        int eq_check    = (lid > 0) ? (k != l_reduced_keys_[lid - 1]) : 0;
        int unique_flag = (eq_check || (lid == 0)) && (gid < n_);

        l_unique_flags_[lid] = unique_flag;
        int unique_id =
            work_group_scan_inclusive_add(it, l_wg_temp_, l_unique_flags_);

        l_unique_ids_[lid] = unique_id;

        if (lid == DIMX_ - 1) l_reduced_block_size_[0] = unique_id;

        for (int off = 1; off < DIMX_; off *= 2) {
            it.barrier();
            int test_unique_id =
                (lid + off < DIMX_) ? l_unique_ids_[lid + off] : ~unique_id;
            eq_check = (unique_id == test_unique_id);
            int update_key =
                eq_check && (lid < (DIMX_ - off)) &&
                ((gid + off) <
                 n_);  // checks if this thread should perform a reduction
            compute_t<To> uval = (update_key) ? l_vals_[lid + off] : init_val;
            it.barrier();
            l_vals_[lid] =
                binOp(l_vals_[lid], uval);  // update if thread requires it
        }

        if (unique_flag) {
            l_reduced_keys_[unique_id - 1] = k;
            l_reduced_vals_[unique_id - 1] = l_vals_[lid];
        }
        it.barrier();

        const int bid = g.get_group_id(0);
        if (lid < l_reduced_block_size_[0]) {
            const int bOffset = bidw * oVInfo_.strides[dims_ordering[3]] +
                                bidz * oVInfo_.strides[dims_ordering[2]] +
                                bidy * oVInfo_.strides[dims_ordering[1]];
            oKeys_[gid] = l_reduced_keys_[lid];
            oVals_[bOffset + (gid)*oVInfo_.strides[DIM_]] =
                l_reduced_vals_[lid];
        }

        reduced_block_sizes_[bid] = l_reduced_block_size_[0];
    }

    int work_group_scan_inclusive_add(sycl::nd_item<3> it,
                                      sycl::local_accessor<int> wg_temp,
                                      sycl::local_accessor<int> arr) const {
        const uint lid = it.get_local_id(0);
        int *active_buf;

        int val    = arr[lid];
        active_buf = arr.get_pointer();

        bool swap_buffer = false;
        for (int off = 1; off <= DIMX_; off *= 2) {
            it.barrier();
            if (lid >= off) { val = val + active_buf[lid - off]; }
            swap_buffer = !swap_buffer;
            active_buf =
                swap_buffer ? wg_temp.get_pointer() : arr.get_pointer();
            active_buf[lid] = val;
        }

        int res = active_buf[lid];
        return res;
    }

   protected:
    sycl::accessor<int> reduced_block_sizes_;
    write_accessor<Tk> oKeys_;
    KParam oKInfo_;
    write_accessor<To> oVals_;
    KParam oVInfo_;
    read_accessor<Tk> iKeys_;
    KParam iKInfo_;
    read_accessor<Ti> iVals_;
    KParam iVInfo_;
    int change_nan_;
    To nanval_;
    int n_;
    int nGroupsZ_;
    int DIMX_;
    int DIM_;
    sycl::local_accessor<Tk> l_keys_;
    sycl::local_accessor<compute_t<To>> l_vals_;
    sycl::local_accessor<Tk> l_reduced_keys_;
    sycl::local_accessor<compute_t<To>> l_reduced_vals_;
    sycl::local_accessor<int> l_unique_ids_;
    sycl::local_accessor<int> l_wg_temp_;
    sycl::local_accessor<int> l_unique_flags_;
    sycl::local_accessor<int> l_reduced_block_size_;
};

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
