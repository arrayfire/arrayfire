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
#include <backend.hpp>
#include <common/err_common.hpp>
#include <af/event.h>

using namespace detail;

af_event_t getEvent(const af_event eventHandle) {
    return *(af_event_t *)eventHandle;
}

af_event getEventHandle(const af_event_t event) {
    af_event_t *eventHandle = new af_event_t;
    *eventHandle            = event;
    return (af_event)eventHandle;
}

af_err af_create_event(af_event *eventHandle) {
    try {
        *eventHandle = makeEventOnActiveQueue();
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_release_event(const af_event eventHandle) {
    try {
        releaseEvent(eventHandle);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_mark_event(const af_event eventHandle) {
    try {
        markEventOnActiveQueue(eventHandle);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_enqueue_wait_event(const af_event eventHandle) {
    try {
        enqueueWaitOnActiveQueue(eventHandle);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_block_event(const af_event eventHandle) {
    try {
        block(eventHandle);
    }
    CATCHALL;

    return AF_SUCCESS;
}
