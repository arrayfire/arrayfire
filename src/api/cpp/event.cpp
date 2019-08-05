/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/event.h>
#include "error.hpp"

#include <utility>

namespace af {

event::event() {
    preserve_ = false;
    AF_THROW(af_create_event(&e_));
}

event::event(af_event e) : e_(e) {}

event::~event() {
    // No dtor throw
    if (!preserve_) { af_release_event(e_); }
}

event::event(event&& other) : e_(nullptr) { *this = std::move(other); }

event& event::operator=(event&& other) {
    af_release_event(this->e_);
    other.unlock();
    this->e_ = other.e_;
    return *this;
}

void event::unlock() { preserve_ = true; }

af_event event::get() const { return e_; }

}  // namespace af
