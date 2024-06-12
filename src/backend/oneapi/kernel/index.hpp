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
#include <common/dispatch.hpp>
#include <debug_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <kernel/assign_kernel_param.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T>
class indexKernel {
    write_accessor<T> out;
    KParam outp;
    read_accessor<T> in;
    KParam inp;
    IndexKernelParam p;
    int nBBS0;
    int nBBS1;

   public:
    indexKernel(write_accessor<T> out_, KParam outp_, read_accessor<T> in_,
                KParam inp_, const IndexKernelParam p_, const int nBBS0_,
                const int nBBS1_)
        : out(out_)
        , outp(outp_)
        , in(in_)
        , inp(inp_)
        , p(p_)
        , nBBS0(nBBS0_)
        , nBBS1(nBBS1_) {}

    int trimIndex(int idx, const int len) const {
        int ret_val = idx;
        if (ret_val < 0) {
            int offset = (abs(ret_val) - 1) % len;
            ret_val    = offset;
        } else if (ret_val >= len) {
            int offset = abs(ret_val) % len;
            ret_val    = len - offset - 1;
        }
        return ret_val;
    }

    void operator()(sycl::nd_item<3> it) const {
        // retrieve index pointers
        // these can be 0 where af_array index is not used
        sycl::group g    = it.get_group();
        const uint* ptr0 = p.ptr[0].get_pointer();
        const uint* ptr1 = p.ptr[1].get_pointer();
        const uint* ptr2 = p.ptr[2].get_pointer();
        const uint* ptr3 = p.ptr[3].get_pointer();
        // retrive booleans that tell us which index to use
        const bool s0 = p.isSeq[0];
        const bool s1 = p.isSeq[1];
        const bool s2 = p.isSeq[2];
        const bool s3 = p.isSeq[3];

        const int gz = g.get_group_id(0) / nBBS0;
        const int gx = g.get_local_range(0) * (g.get_group_id(0) - gz * nBBS0) +
                       it.get_local_id(0);

        const int gw =
            (g.get_group_id(1) + g.get_group_id(2) * g.get_group_range(1)) /
            nBBS1;
        const int gy =
            g.get_local_range(1) * ((g.get_group_id(1) +
                                     g.get_group_id(2) * g.get_group_range(1)) -
                                    gw * nBBS1) +
            it.get_local_id(1);

        size_t odims0 = outp.dims[0];
        size_t odims1 = outp.dims[1];
        size_t odims2 = outp.dims[2];
        size_t odims3 = outp.dims[3];

        if (gx < odims0 && gy < odims1 && gz < odims2 && gw < odims3) {
            // calculate pointer offsets for input
            int i = p.strds[0] *
                    trimIndex(s0 ? gx + p.offs[0] : ptr0[gx], inp.dims[0]);
            int j = p.strds[1] *
                    trimIndex(s1 ? gy + p.offs[1] : ptr1[gy], inp.dims[1]);
            int k = p.strds[2] *
                    trimIndex(s2 ? gz + p.offs[2] : ptr2[gz], inp.dims[2]);
            int l = p.strds[3] *
                    trimIndex(s3 ? gw + p.offs[3] : ptr3[gw], inp.dims[3]);
            // offset input and output pointers
            const T* src = (const T*)in.get_pointer() + (i + j + k + l);
            T* dst       = (T*)out.get_pointer() +
                     (gx * outp.strides[0] + gy * outp.strides[1] +
                      gz * outp.strides[2] + gw * outp.strides[3]);
            // set the output
            dst[0] = src[0];
        }
    }
};

template<typename T>
void index(Param<T> out, Param<T> in, IndexKernelParam& p,
           std::vector<Array<uint>>& idxArrs) {
    sycl::range<3> threads(0, 0, 1);
    switch (out.info.dims[1]) {
        case 1: threads[1] = 1; break;
        case 2: threads[1] = 2; break;
        case 3:
        case 4: threads[1] = 4; break;
        default: threads[1] = 8; break;
    }
    threads[0] = static_cast<unsigned>(256.f / threads[1]);

    int blks_x = divup(out.info.dims[0], threads[0]);
    int blks_y = divup(out.info.dims[1], threads[1]);

    sycl::range<3> blocks(blks_x * out.info.dims[2], blks_y * out.info.dims[3],
                          1);

    const size_t maxBlocksY =
        getDevice().get_info<sycl::info::device::max_work_item_sizes<3>>()[2];
    blocks[2] = divup(blocks[1], maxBlocksY);
    blocks[1] = divup(blocks[1], blocks[2]) * threads[1];
    blocks[1] = blocks[1] * threads[1];
    blocks[0] *= threads[0];

    sycl::nd_range<3> marange(blocks, threads);
    for (dim_t x = 0; x < 4; ++x) {
        auto idxArrs_get = idxArrs[x].get();
        getQueue().submit([&](sycl::handler& h) {
            auto pp = p;
            pp.ptr[x] =
                idxArrs_get->get_access<sycl::access::mode::read>(h);
    
            h.parallel_for(
                marange,
                indexKernel<T>(
                    out.data->template get_access<sycl::access::mode::write>(h),
                    out.info,
                    in.data->template get_access<sycl::access::mode::read>(h),
                    in.info, pp, blks_x, blks_y));
        });
    }
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
