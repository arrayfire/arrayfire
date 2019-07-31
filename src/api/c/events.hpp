/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <backend.hpp>
#include <Event.hpp>
#include <af/event.h>

typedef struct {
  /// Underlying EventBase<T>
  detail::Event *event;
} af_event_t;

af_event getEventHandle(const af_event_t event);

af_event_t getEvent(const af_event eventHandle);
