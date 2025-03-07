/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <sycl/sycl.hpp>

#include <kernel/KParam.hpp>
#include <types.hpp>
#include <af/dim4.hpp>

#include <optional>

namespace arrayfire {
namespace oneapi {

template<typename T>
struct Param {
    sycl::buffer<T>* data;
    KParam info;
    Param& operator=(const Param& other) = default;
    Param(const Param& other)            = default;
    Param(Param&& other)                 = default;

    dim_t* dims_ptr() { return info.dims; }
    dim_t* strides_ptr() { return info.strides; }

    // AF_DEPRECATED("Use Array<T>")
    Param() : data(nullptr), info{{0, 0, 0, 0}, {0, 0, 0, 0}, 0} {}

    // AF_DEPRECATED("Use Array<T>")
    Param(sycl::buffer<T>* data_, KParam info_) : data(data_), info(info_) {}

    template<sycl::access::mode MODE>
    sycl::accessor<data_t<T>, 1, MODE> get_accessor(sycl::handler& h) const {
        auto o = data->template reinterpret<data_t<T>>();
        return sycl::accessor<data_t<T>, 1, MODE>(o, h);
    }

    ~Param() = default;
};

template<typename T, sycl::access_mode AM>
struct AParam {
    sycl::accessor<T, 1, AM, sycl::target::device,
                   sycl::access::placeholder::true_t>
        data;
    af::dim4 dims;
    af::dim4 strides;
    dim_t offset;
    AParam& operator=(const AParam& other) = default;
    AParam(const AParam& other)            = default;
    AParam(AParam&& other)                 = default;

    dim_t* dims_ptr() { return dims.get(); }
    dim_t* strides_ptr() { return strides.get(); }

    // AF_DEPRECATED("Use Array<T>")
    AParam() : data(), dims{0, 0, 0, 0}, strides{0, 0, 0, 0}, offset(0) {}

    AParam(sycl::buffer<T, 1>& data_, const dim_t dims_[4],
           const dim_t strides_[4], dim_t offset_)
        : data(data_), dims(4, dims_), strides(4, strides_), offset(offset_) {}
    // AF_DEPRECATED("Use Array<T>")
    AParam(sycl::handler& h, sycl::buffer<T, 1>& data_, const dim_t dims_[4],
           const dim_t strides_[4], dim_t offset_)
        : data(data_), dims(4, dims_), strides(4, strides_), offset(offset_) {
        require(h);
    }

    template<sycl::access::mode MODE>
    sycl::accessor<data_t<T>, 1, MODE> get_accessor(sycl::handler& h) const {
        return *data;
    }

    void require(sycl::handler& h) const { h.require(data); }

    operator KParam() const {
        return KParam{{dims[0], dims[1], dims[2], dims[3]},
                      {strides[0], strides[1], strides[2], strides[3]},
                      offset};
    }

    ~AParam() = default;
};

// AF_DEPRECATED("Use Array<T>")
template<typename T>
Param<T> makeParam(sycl::buffer<T>& mem, int off, const int dims[4],
                   const int strides[4]);

namespace opencl {

template<typename T>
struct Param {
    cl_mem data;
    KParam info;
    Param& operator=(const Param& other) = default;
    Param(const Param& other)            = default;
    Param(Param&& other)                 = default;
    Param(cl_mem data_, KParam info_) : data(data_), info(info_) {}

    // AF_DEPRECATED("Use Array<T>")
    Param() : data(nullptr), info{{0, 0, 0, 0}, {0, 0, 0, 0}, 0} {}

    // AF_DEPRECATED("Use Array<T>")
    Param(sycl::buffer<T>* data_, KParam info_) : data(data_), info(info_) {}

    ~Param() = default;
};
}  // namespace opencl

}  // namespace oneapi
}  // namespace arrayfire
