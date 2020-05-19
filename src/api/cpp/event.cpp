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

namespace af {

event::event() : e_{} { AF_THROW(af_create_event(&e_)); }

event::event(af_event e) : e_(e) {}

event::~event() {
    // No dtor throw
    if (e_) { af_delete_event(e_); }
}

// NOLINTNEXTLINE(performance-noexcept-move-constructor) we can't change the API
event::event(event&& other) : e_(other.e_) { other.e_ = 0; }

// NOLINTNEXTLINE(performance-noexcept-move-constructor) we can't change the API
event& event::operator=(event&& other) {
    af_delete_event(this->e_);
    this->e_ = other.e_;
    other.e_ = 0;
    return *this;
}

af_event event::get() const { return e_; }

void event::mark() { AF_THROW(af_mark_event(e_)); }

void event::enqueue() { AF_THROW(af_enqueue_wait_event(e_)); }

void event::block() const { AF_THROW(af_block_event(e_)); }

}  // namespace af
