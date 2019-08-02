/*******************************************************
 * Copyright (c) 2014, ArrayFire
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

TEST(EventTests, SimpleCreateRelease) {
  af_event event;
  ASSERT_SUCCESS(af_create_event(&event));
  ASSERT_SUCCESS(af_release_event(event));
}


TEST(EventTests, MarkEnqueueAndBlock) {
  af_event event;
  ASSERT_SUCCESS(af_create_event(&event));

  ASSERT_SUCCESS(af_mark_event(event));
  ASSERT_SUCCESS(af_enqueue_wait_event(event));
  ASSERT_SUCCESS(af_block_event(event));
  
  ASSERT_SUCCESS(af_release_event(event));
}
