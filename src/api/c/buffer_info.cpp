/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memoryapi.hpp>

#include <backend.hpp>
#include <common/err_common.hpp>
#include <af/device.h>
#include <af/event.h>
#include <af/memory.h>

BufferInfo &getBufferInfo(const af_buffer_info handle) {
    return *(BufferInfo *)handle;
}

af_buffer_info getHandle(BufferInfo &buf) {
    BufferInfo *handle;
    handle = &buf;
    return (af_buffer_info)handle;
}

detail::Event &getEventFromBufferInfoHandle(const af_buffer_info handle) {
    return getEvent(getBufferInfo(handle).event);
}

af_err af_create_buffer_info(af_buffer_info *handle, void *ptr,
                             af_event event) {
    try {
        BufferInfo *buf = new BufferInfo({ptr, event});
        *handle         = getHandle(*((BufferInfo *)buf));
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_delete_buffer_info(af_buffer_info handle) {
    try {
        /// NB: deleting a memory event buf does frees the associated memory
        /// and deletes the associated event. Use unlock functions to free
        /// resources individually
        BufferInfo &buf = getBufferInfo(handle);
        af_release_event(buf.event);
        if (buf.ptr) { af_free_device(buf.ptr); }

        delete (BufferInfo *)handle;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_buffer_info_get_ptr(void **ptr, af_buffer_info handle) {
    try {
        BufferInfo &buf = getBufferInfo(handle);
        *ptr            = buf.ptr;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_buffer_info_get_event(af_event *event, af_buffer_info handle) {
    try {
        BufferInfo &buf = getBufferInfo(handle);
        *event          = buf.event;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_buffer_info_set_ptr(af_buffer_info handle, void *ptr) {
    try {
        BufferInfo &buf = getBufferInfo(handle);
        buf.ptr         = ptr;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_buffer_info_set_event(af_buffer_info handle, af_event event) {
    try {
        BufferInfo &buf = getBufferInfo(handle);
        buf.event       = event;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_unlock_buffer_info_event(af_event *event, af_buffer_info handle) {
    try {
        af_buffer_info_get_event(event, handle);
        BufferInfo &buf = getBufferInfo(handle);
        buf.event       = 0;
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_unlock_buffer_info_ptr(void **ptr, af_buffer_info handle) {
    try {
        af_buffer_info_get_ptr(ptr, handle);
        BufferInfo &buf = getBufferInfo(handle);
        buf.ptr         = 0;
    }
    CATCHALL;

    return AF_SUCCESS;
}
