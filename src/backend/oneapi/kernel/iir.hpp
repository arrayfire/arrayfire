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
#include <math.hpp>

#include <sycl/sycl.hpp>

namespace arrayfire {
namespace oneapi {
namespace kernel {

template<typename T, bool batch_a>
class iirKernel {
   public:
    iirKernel(write_accessor<T> y, KParam yInfo, read_accessor<T> c,
              KParam cInfo, read_accessor<T> a, KParam aInfo,
              sycl::local_accessor<T> s_z, sycl::local_accessor<T> s_a,
              sycl::local_accessor<T> s_y, int groups_y)
        : y_(y)
        , yInfo_(yInfo)
        , c_(c)
        , cInfo_(cInfo)
        , a_(a)
        , aInfo_(aInfo)
        , s_z_(s_z)
        , s_a_(s_a)
        , s_y_(s_y)
        , groups_y_(groups_y) {}

    void operator()(sycl::nd_item<2> it) const {
        sycl::group g = it.get_group();

        const int idz = g.get_group_id(0);
        const int idw = g.get_group_id(1) / groups_y_;
        const int idy = g.get_group_id(1) - idw * groups_y_;

        const int tx    = it.get_local_id(0);
        const int num_a = aInfo_.dims[0];

        int y_off = idw * yInfo_.strides[3] + idz * yInfo_.strides[2] +
                    idy * yInfo_.strides[1];
        int c_off = idw * cInfo_.strides[3] + idz * cInfo_.strides[2] +
                    idy * cInfo_.strides[1];
        int a_off = 0;

        if (batch_a)
            a_off = idw * aInfo_.strides[3] + idz * aInfo_.strides[2] +
                    idy * aInfo_.strides[1];

        T *d_y       = y_.get_pointer() + y_off;
        const T *d_c = c_.get_pointer() + c_off;
        const T *d_a = a_.get_pointer() + a_off;
        const int repeat =
            (num_a + g.get_local_range(0) - 1) / g.get_local_range(0);

        for (int ii = tx; ii < num_a; ii += g.get_local_range(0)) {
            s_z_[ii] = scalar<T>(0);
            s_a_[ii] = (ii < num_a) ? d_a[ii] : scalar<T>(0);
        }
        group_barrier(g);

        for (int i = 0; i < yInfo_.dims[0]; i++) {
            if (tx == 0) {
                s_y_[0] = (d_c[i] + s_z_[0]) / s_a_[0];
                d_y[i]  = s_y_[0];
            }
            group_barrier(g);

            for (int ii = 0; ii < repeat; ii++) {
                int id = ii * g.get_local_range(0) + tx + 1;

                T z;

                if (id < num_a) {
                    z = s_z_[id] - s_a_[id] * s_y_[0];
                } else {
                    z = scalar<T>(0);
                }
                group_barrier(g);

                if ((id - 1) < num_a) { s_z_[id - 1] = z; }
                group_barrier(g);
            }
        }
    }

   protected:
    write_accessor<T> y_;
    KParam yInfo_;
    read_accessor<T> c_;
    KParam cInfo_;
    read_accessor<T> a_;
    KParam aInfo_;
    sycl::local_accessor<T> s_z_;
    sycl::local_accessor<T> s_a_;
    sycl::local_accessor<T> s_y_;
    int groups_y_;
};

template<typename T, bool batch_a>
void iir(Param<T> y, Param<T> c, Param<T> a) {
    const size_t groups_y = y.info.dims[1];
    const size_t groups_x = y.info.dims[2];

    size_t threads = 256;
    while (threads > y.info.dims[0] && threads > 32) threads /= 2;
    sycl::range<2> local = sycl::range{threads, 1};

    sycl::range<2> global =
        sycl::range<2>{groups_x * local[0], groups_y * y.info.dims[3]};

    getQueue().submit([&](sycl::handler &h) {
        write_accessor<T> yAcc{*y.data, h};
        read_accessor<T> cAcc{*c.data, h};
        read_accessor<T> aAcc{*a.data, h};

        unsigned num_a = a.info.dims[0];

        auto s_z = sycl::local_accessor<T>(num_a, h);
        auto s_a = sycl::local_accessor<T>(num_a, h);
        auto s_y = sycl::local_accessor<T>(1, h);

        if (batch_a) {
            h.parallel_for(sycl::nd_range{global, local},
                           iirKernel<T, true>(yAcc, y.info, cAcc, c.info, aAcc,
                                              a.info, s_z, s_a, s_y, groups_y));
        } else {
            h.parallel_for(
                sycl::nd_range{global, local},
                iirKernel<T, false>(yAcc, y.info, cAcc, c.info, aAcc, a.info,
                                    s_z, s_a, s_y, groups_y));
        }
    });
    ONEAPI_DEBUG_FINISH(getQueue());
}

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
