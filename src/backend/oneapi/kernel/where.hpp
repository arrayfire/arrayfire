/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Param.hpp>
#include <backend.hpp>
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <err_oneapi.hpp>
#include <kernel/default_config.hpp>
#include <kernel/scan_first.hpp>
#include <memory.hpp>

#include <Param.hpp>
#include <backend.hpp>
#include <math.hpp>

namespace oneapi {
namespace kernel {

template<typename T>
using read_accessor = sycl::accessor<T, 1, sycl::access::mode::read>;

template<typename T>
using write_accessor = sycl::accessor<T, 1, sycl::access::mode::write>;

template<typename T>
class whereKernel {
   public:
    whereKernel(write_accessor<uint> out_acc, KParam oInfo,
                read_accessor<uint> otmp_acc, KParam otInfo,
                read_accessor<uint> rtmp_acc, KParam rtInfo,
                read_accessor<T> in_acc, KParam iInfo, uint groups_x,
                uint groups_y, uint lim, sycl::stream debug)
        : out_acc_(out_acc)
        , oInfo_(oInfo)
        , otmp_acc_(otmp_acc)
        , otInfo_(otInfo)
        , rtmp_acc_(rtmp_acc)
        , rtInfo_(rtInfo)
        , in_acc_(in_acc)
        , iInfo_(iInfo)
        , groups_x_(groups_x)
        , groups_y_(groups_y)
        , lim_(lim)
        , debug_(debug) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g   = it.get_group();
        const uint lidx = it.get_local_id(0);
        const uint lidy = it.get_local_id(1);

        const uint zid       = g.get_group_id(0) / groups_x_;
        const uint wid       = g.get_group_id(1) / groups_y_;
        const uint groupId_x = g.get_group_id(0) - (groups_x_)*zid;
        const uint groupId_y = g.get_group_id(1) - (groups_y_)*wid;
        const uint xid       = groupId_x * g.get_local_range(0) * lim_ + lidx;
        const uint yid       = groupId_y * g.get_local_range(1) + lidy;

        const uint *otptr = otmp_acc_.get_pointer();
        const uint *rtptr = rtmp_acc_.get_pointer();
        const T *iptr     = in_acc_.get_pointer();

        const uint off = wid * otInfo_.strides[3] + zid * otInfo_.strides[2] +
                         yid * otInfo_.strides[1];
        const uint bid = wid * rtInfo_.strides[3] + zid * rtInfo_.strides[2] +
                         yid * rtInfo_.strides[1] + groupId_x;

        otptr += wid * otInfo_.strides[3] + zid * otInfo_.strides[2] +
                 yid * otInfo_.strides[1];
        iptr += wid * iInfo_.strides[3] + zid * iInfo_.strides[2] +
                yid * iInfo_.strides[1];

        bool cond = (yid < otInfo_.dims[1]) && (zid < otInfo_.dims[2]) &&
                    (wid < otInfo_.dims[3]);
        T zero = scalar<T>(0);

        if (!cond) return;

        uint accum = (bid == 0) ? 0 : rtptr[bid - 1];

        for (uint k = 0, id = xid; k < lim_ && id < otInfo_.dims[0];
             k++, id += g.get_local_range(0)) {
            uint idx = otptr[id] + accum;
            if (iptr[id] != zero) out_acc_[idx - 1] = (off + id);
        }
    }

   protected:
    write_accessor<uint> out_acc_;
    read_accessor<uint> otmp_acc_;
    read_accessor<uint> rtmp_acc_;
    read_accessor<T> in_acc_;
    KParam oInfo_, otInfo_, rtInfo_, iInfo_;
    uint groups_x_, groups_y_, lim_;
    sycl::stream debug_;
};

template<typename T>
static void where(Param<uint> &out, Param<T> in) {
    uint threads_x = nextpow2(std::max(32u, (uint)in.info.dims[0]));
    threads_x      = std::min(threads_x, THREADS_PER_BLOCK);
    uint threads_y = THREADS_PER_BLOCK / threads_x;

    uint groups_x = divup((uint)in.info.dims[0], (uint)(threads_x * REPEAT));
    uint groups_y = divup(in.info.dims[1], threads_y);

    Param<uint> rtmp;
    Param<uint> otmp;
    rtmp.info.dims[0]    = groups_x;
    otmp.info.dims[0]    = in.info.dims[0];
    rtmp.info.strides[0] = 1;
    otmp.info.strides[0] = 1;

    for (int k = 1; k < 4; k++) {
        rtmp.info.dims[k]    = in.info.dims[k];
        rtmp.info.strides[k] = rtmp.info.strides[k - 1] * rtmp.info.dims[k - 1];

        otmp.info.dims[k]    = in.info.dims[k];
        otmp.info.strides[k] = otmp.info.strides[k - 1] * otmp.info.dims[k - 1];
    }

    uintl rtmp_elements = rtmp.info.strides[3] * rtmp.info.dims[3];
    uintl otmp_elements = otmp.info.strides[3] * otmp.info.dims[3];
    auto rtmp_alloc     = memAlloc<uint>(rtmp_elements);
    auto otmp_alloc     = memAlloc<uint>(otmp_elements);
    rtmp.data           = rtmp_alloc.get();
    otmp.data           = otmp_alloc.get();

    scan_first_launcher<T, uint, af_notzero_t>(
        otmp, rtmp, in, groups_x, groups_y, threads_x, false, true);

    // Linearize the dimensions and perform scan
    Param<uint> ltmp  = rtmp;
    ltmp.info.dims[0] = rtmp_elements;
    for (int k = 1; k < 4; k++) {
        ltmp.info.dims[k]    = 1;
        ltmp.info.strides[k] = rtmp_elements;
    }

    scan_first<uint, uint, af_add_t>(ltmp, ltmp, true);

    // Get output size and allocate output
    uint total;
    sycl::buffer retBuffer(&total, {1},
                           {sycl::property::buffer::use_host_ptr()});

    getQueue()
        .submit([&](sycl::handler &h) {
            auto acc_in  = rtmp.data->get_access(h, sycl::range{1},
                                                 sycl::id{rtmp_elements - 1});
            auto acc_out = retBuffer.get_access();
            h.copy(acc_in, acc_out);
        })
        .wait();

    auto out_alloc = memAlloc<uint>(total);
    out.data       = out_alloc.get();

    out.info.dims[0]    = total;
    out.info.strides[0] = 1;
    for (int k = 1; k < 4; k++) {
        out.info.dims[k]    = 1;
        out.info.strides[k] = total;
    }

    sycl::range<2> local(threads_x, THREADS_PER_BLOCK / threads_x);
    sycl::range<2> global(groups_x * in.info.dims[2] * local[0],
                          groups_y * in.info.dims[3] * local[1]);
    uint lim = divup(otmp.info.dims[0], (threads_x * groups_x));

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<uint> out_acc{*out.data, h};
        read_accessor<uint> otmp_acc{*otmp.data, h};
        read_accessor<uint> rtmp_acc{*rtmp.data, h};
        read_accessor<T> in_acc{*in.data, h};

        sycl::stream debug_stream(2048 * 256, 128, h);
        h.parallel_for(sycl::nd_range<2>(global, local),
                       whereKernel<T>(out_acc, out.info, otmp_acc, otmp.info,
                                      rtmp_acc, rtmp.info, in_acc, in.info,
                                      groups_x, groups_y, lim, debug_stream));
    });
    ONEAPI_DEBUG_FINISH(getQueue());
    out_alloc.release();
}

}  // namespace kernel
}  // namespace oneapi
