/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>

#if AF_API_VERSION >= 37

typedef void* af_event;

#ifdef __cplusplus
namespace af {

/// A simple RAII wrapper for af_event
class AFAPI event {
    af_event e_;
    bool preserve_;  // Preserve the event after wrapper deletion
   public:
    event(af_event e);
#if AF_COMPILER_CXX_RVALUE_REFERENCES
    event(event&& other);
    event& operator=(event&& other);
#endif
    event();
    ~event();
    void unlock();
    af_event get() const;

   private:
    event& operator=(const event& other);
    event(const event& other);
};

}  // namespace af
#endif

#ifdef __cplusplus
extern "C" {
#endif

AFAPI af_err af_create_event(af_event* eventHandle);

AFAPI af_err af_release_event(const af_event eventHandle);

AFAPI af_err af_mark_event(const af_event eventHandle);

AFAPI af_err af_enqueue_wait_event(const af_event eventHandle);

AFAPI af_err af_block_event(const af_event eventHandle);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // AF_API_VERSION >= 37
