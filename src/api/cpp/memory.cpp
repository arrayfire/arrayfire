/*******************************************************
 * Copyright (c) 2014, ArrayFire
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

memory_event_pair::memory_event_pair(void* ptr, af_event event) {
    AF_CHECK(af_create_memory_event_pair(&p_, ptr, event));
}

memory_event_pair::memory_event_pair(af_memory_event_pair p) : p_(p) {}

memory_event_pair::memory_event_pair(memory_event_pair&& other) : p_(nullptr) {
    *this = std::move(other);
}

memory_event_pair& memory_event_pair::operator=(memory_event_pair&& other) {
    af_release_memory_event_pair(this->p_);
    other.unlock();
    this->p_ = other.p_;
    return *this;
}

memory_event_pair::~memory_event_pair() {
    // No throw dtor
    if (!preserve_) { af_release_memory_event_pair(p_); }
}

void memory_event_pair::unlock() { preserve_ = true; }

void* memory_event_pair::getPtr() const {
    void* ptr;
    AF_CHECK(af_memory_event_pair_get_ptr(&ptr, p_));
    return ptr;
}

void memory_event_pair::setPtr(void* ptr) const {
    AF_CHECK(af_memory_event_pair_set_ptr(p_, ptr));
}

af_event memory_event_pair::getEvent() const {
    af_event e;
    AF_CHECK(af_memory_event_pair_get_event(&e, p_));
    return e;
}

af_memory_event_pair memory_event_pair::get() const { return p_; }

}  // namespace af
