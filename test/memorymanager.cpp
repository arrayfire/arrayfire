/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/event.h>
#include <af/memory.h>
#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>

TEST(EventTests, SimpleCreateRelease) {
  af_event event;
  ASSERT_SUCCESS(af_create_event(&event));
  af_memory_event_pair pair;
  ASSERT_SUCCESS(af_create_memory_event_pair(&pair, NULL, event));

  ASSERT_SUCCESS(af_release_memory_event_pair(pair));
  ASSERT_SUCCESS(af_release_event(event));
}

TEST(EventTests, EventAndPtrAttributes) {
  af_event event;
  ASSERT_SUCCESS(af_create_event(&event));
  void *ptr;
  af_memory_event_pair pair;
  ASSERT_SUCCESS(af_create_memory_event_pair(&pair, ptr, event));
  af_event anEvent;
  ASSERT_SUCCESS(af_memory_event_pair_get_event(pair, &anEvent));
  ASSERT_EQ(event, anEvent);
  void *somePtr;
  ASSERT_SUCCESS(af_memory_event_pair_get_ptr(pair, &somePtr));
  ASSERT_EQ(ptr, somePtr);

  af_event anotherEvent;
  ASSERT_SUCCESS(af_create_event(&anotherEvent));  
  ASSERT_SUCCESS(af_memory_event_pair_set_event(pair, anotherEvent));
  af_event yetAnotherEvent;
  ASSERT_SUCCESS(af_memory_event_pair_get_event(pair, &yetAnotherEvent));
  ASSERT_NE(yetAnotherEvent, event);
  ASSERT_EQ(yetAnotherEvent, anotherEvent);

  void* anotherPtr;
  ASSERT_SUCCESS(af_memory_event_pair_set_ptr(pair, anotherPtr));
  void* yetAnotherPtr;
  ASSERT_SUCCESS(af_memory_event_pair_get_ptr(pair, &yetAnotherPtr));
  ASSERT_NE(yetAnotherPtr, ptr);
  ASSERT_EQ(yetAnotherPtr, anotherPtr);

  ASSERT_SUCCESS(af_release_memory_event_pair(pair));
  ASSERT_SUCCESS(af_release_event(event));
  ASSERT_SUCCESS(af_release_event(anotherEvent));
}
