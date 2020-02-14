/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/event.h>

#include <memory>
#include <utility>

#include <iostream>

using af::event;

TEST(EventTests, SimpleCreateRelease) {
    af_event event;
    ASSERT_SUCCESS(af_create_event(&event));
    ASSERT_SUCCESS(af_delete_event(event));
}

TEST(EventTests, MarkEnqueueAndBlock) {
    af_event event;
    ASSERT_SUCCESS(af_create_event(&event));
    ASSERT_SUCCESS(af_mark_event(event));
    ASSERT_SUCCESS(af_enqueue_wait_event(event));
    ASSERT_SUCCESS(af_block_event(event));
    ASSERT_SUCCESS(af_delete_event(event));
}

TEST(EventTests, EventCreateAndMove) {
    af_event eventHandle;
    ASSERT_SUCCESS(af_create_event(&eventHandle));

    event e(eventHandle);
    e.mark();
    ASSERT_EQ(eventHandle, e.get());

    auto otherEvent = std::move(e);
    ASSERT_EQ(otherEvent.get(), eventHandle);

    event f;
    af_event fE        = f.get();
    event anotherEvent = std::move(f);
    ASSERT_EQ(fE, anotherEvent.get());
    af::sync();
}
