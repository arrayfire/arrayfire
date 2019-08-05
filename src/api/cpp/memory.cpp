/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <af/event.h>
#include <af/memory.h>

#include <utility>

namespace af {

buffer_info::buffer_info(void* ptr, af_event event) {
    preserve_ = false;
    AF_CHECK(af_create_buffer_info(&p_, ptr, event));
}

buffer_info::buffer_info(af_buffer_info p) : p_(p) {}

buffer_info::buffer_info(buffer_info&& other) : p_(nullptr) {
    *this = std::move(other);
}

buffer_info& buffer_info::operator=(buffer_info&& other) {
    af_release_buffer_info(this->p_);
    other.unlock();
    this->p_ = other.p_;
    return *this;
}

buffer_info::~buffer_info() {
    // No throw dtor
    if (!preserve_) { af_release_buffer_info(p_); }
}

void buffer_info::unlock() { preserve_ = true; }

void* buffer_info::getPtr() const {
    void* ptr;
    AF_CHECK(af_buffer_info_get_ptr(&ptr, p_));
    return ptr;
}

af_event buffer_info::getEvent() const {
    af_event e;
    AF_CHECK(af_buffer_info_get_event(&e, p_));
    return e;
}

af_buffer_info buffer_info::get() const { return p_; }

}  // namespace af
