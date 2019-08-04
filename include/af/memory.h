/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>
#include <af/event.h>

#if AF_API_VERSION >= 37

typedef void* af_memory_event_pair;

#ifdef __cplusplus
namespace af {

/// A simple RAII wrapper for af_memory_event_pair
class memory_event_pair {
    af_memory_event_pair p_;
    bool preserve_ = false;  // Preserve the event after wrapper deletion
   public:
    memory_event_pair(af_memory_event_pair p);
    memory_event_pair(void* ptr, af_event event);
    ~memory_event_pair();
#if AF_COMPILER_CXX_RVALUE_REFERENCES
    memory_event_pair(memory_event_pair&& other);
    memory_event_pair& operator=(memory_event_pair&& other);
#endif
    void unlock();
    void* getPtr() const;
    void setPtr(void* ptr) const;
    af_event getEvent() const;
    af_memory_event_pair get() const;

   private:
    memory_event_pair& operator=(const memory_event_pair& other);
    memory_event_pair(const memory_event_pair& other);
};

}  // namespace af
#endif

#ifdef __cplusplus
extern "C" {
#endif

AFAPI af_err af_create_memory_event_pair(af_memory_event_pair* pair, void* ptr,
                                         af_event event);

AFAPI af_err af_release_memory_event_pair(af_memory_event_pair pair);

AFAPI af_err af_memory_event_pair_set_ptr(af_memory_event_pair pair, void* ptr);

AFAPI af_err af_memory_event_pair_set_event(af_memory_event_pair pairHandle,
                                            af_event event);

AFAPI af_err af_memory_event_pair_get_ptr(void** ptr,
                                          af_memory_event_pair pair);

AFAPI af_err af_memory_event_pair_get_event(af_event* event,
                                            af_memory_event_pair pair);

#ifdef __cplusplus
}
#endif

#endif
