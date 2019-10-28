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

namespace af {

buffer_info::buffer_info(void* ptr, af_event event) {
    AF_CHECK(af_create_buffer_info(&p_, ptr, event));
}

buffer_info::buffer_info(af_buffer_info p) : p_(p) {}

buffer_info::buffer_info(buffer_info&& other) : p_(other.p_) { other.p_ = 0; }

buffer_info& buffer_info::operator=(buffer_info&& other) {
    af_delete_buffer_info(this->p_);
    this->p_ = other.p_;
    other.p_ = 0;
    return *this;
}

buffer_info::~buffer_info() {
    // No throw dtor
    af_delete_buffer_info(p_);
}

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

void buffer_info::setPtr(void* ptr) {
    AF_CHECK(af_buffer_info_set_ptr(p_, ptr));
}

void buffer_info::setEvent(af_event event) {
    AF_CHECK(af_buffer_info_set_event(p_, event));
}

af_event buffer_info::unlockEvent() {
    af_event event;
    AF_CHECK(af_unlock_buffer_info_event(&event, p_));
    // Zero out the event
    AF_CHECK(af_buffer_info_set_event(p_, 0));
    return event;
}

void* buffer_info::unlockPtr() {
    void* ptr;
    AF_CHECK(af_unlock_buffer_info_ptr(&ptr, p_));
    // Zero out the ptr
    AF_CHECK(af_buffer_info_set_ptr(p_, 0));
    return ptr;
}

af_buffer_info buffer_info::get() const { return p_; }

}  // namespace af
