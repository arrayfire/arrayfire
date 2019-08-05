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

typedef void* af_buffer_info;

#ifdef __cplusplus
namespace af {

/// A simple RAII wrapper for af_buffer_info
class buffer_info {
    af_buffer_info p_;
    bool preserve_ = false;  // Preserve the event after wrapper deletion
   public:
    buffer_info(af_buffer_info p);
    buffer_info(void* ptr, af_event event);
    ~buffer_info();
#if AF_COMPILER_CXX_RVALUE_REFERENCES
    buffer_info(buffer_info&& other);
    buffer_info& operator=(buffer_info&& other);
#endif
    void unlock();
    void* getPtr() const;
    void setPtr(void* ptr) const;
    af_event getEvent() const;
    af_buffer_info get() const;

   private:
    buffer_info& operator=(const buffer_info& other);
    buffer_info(const buffer_info& other);
};

}  // namespace af
#endif

#ifdef __cplusplus
extern "C" {
#endif

AFAPI af_err af_create_buffer_info(af_buffer_info* pair, void* ptr,
                                   af_event event);

AFAPI af_err af_release_buffer_info(af_buffer_info pair);

AFAPI af_err af_buffer_info_get_ptr(void** ptr, af_buffer_info pair);

AFAPI af_err af_buffer_info_get_event(af_event* event, af_buffer_info pair);

#ifdef __cplusplus
}
#endif

#endif
