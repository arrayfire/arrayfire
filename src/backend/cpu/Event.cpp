/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Event.hpp>
#include <queue.hpp>

namespace cpu {
/// \brief Creates a new event and marks it in the queue
Event make_event(cpu::queue &queue) {
    Event e;
    if (0 == e.create()) { e.mark(queue); }
    return e;
}
}  // namespace cpu
