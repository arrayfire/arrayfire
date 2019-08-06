/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <events.hpp>

#include <Event.hpp>
#include <common/err_common.hpp>
#include <af/device.h>
#include <af/event.h>

using namespace detail;

Event &getEvent(const af_event handle) {
    Event &event = *(Event *)handle;
    return event;
}

af_event getHandle(const Event &event) { return (af_event)&event; }

af_err af_create_event(af_event *handle) {
    try {
        AF_CHECK(af_init());
        *handle = createEvent();
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_release_event(const af_event handle) {
    try {
        releaseEvent(handle);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_mark_event(const af_event handle) {
    try {
        markEventOnActiveQueue(handle);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_enqueue_wait_event(const af_event handle) {
    try {
        enqueueWaitOnActiveQueue(handle);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_block_event(const af_event handle) {
    try {
        block(handle);
    }
    CATCHALL;

    return AF_SUCCESS;
}
