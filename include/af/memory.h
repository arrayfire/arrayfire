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
#include <af/event.h>

#if AF_API_VERSION >= 37

typedef void* af_buffer_info;

typedef void* af_memory_manager;

#ifdef __cplusplus
namespace af {

/// A simple RAII wrapper for af_buffer_info
class AFAPI buffer_info {
    af_buffer_info p_;

   public:
    buffer_info(af_buffer_info p);
    buffer_info(void* ptr, af_event event);
    ~buffer_info();
#if AF_COMPILER_CXX_RVALUE_REFERENCES
    buffer_info(buffer_info&& other);
    buffer_info& operator=(buffer_info&& other);
#endif
    void* getPtr() const;
    void setPtr(void* ptr);
    af_event getEvent() const;
    void setEvent(af_event event);
    af_buffer_info get() const;
    af_event unlockEvent();
    void* unlockPtr();

   private:
    buffer_info& operator=(const buffer_info& other);
    buffer_info(const buffer_info& other);
};

}  // namespace af
#endif

#ifdef __cplusplus
extern "C" {
#endif

AFAPI af_err af_create_buffer_info(af_buffer_info* buf, void* ptr,
                                   af_event event);

/// \brief deletes the \ref af_buffer_info and the resources its tracking
///
///  Deletes the \ref af_buffer_info object and its tracked resources. If buffer
///  still holds
/// the pointer,  that pointer is freed after its associated event has
/// triggered.
///
/// \param[in] buf The af_buffer_info object that will be deleted
/// \returns AF_SUCCESS
AFAPI af_err af_delete_buffer_info(af_buffer_info buf);

AFAPI af_err af_buffer_info_get_ptr(void** ptr, af_buffer_info buf);

AFAPI af_err af_buffer_info_get_event(af_event* event, af_buffer_info buf);

AFAPI af_err af_buffer_info_set_ptr(af_buffer_info buf, void* ptr);

AFAPI af_err af_buffer_info_set_event(af_buffer_info buf, af_event event);

/// \brief Disassociates the \ref af_event from the \ref af_buffer_info object
///
/// Gets the \ref af_event and disassociated it from the af_buffer_info object.
/// Deleting the af_buffer_info object will not affect this event.
///
/// param[out] event The \ref af_event that will be disassociated. If NULL no
/// event is
///                   returned and the event is NOT freed
/// param[in] buf The target \ref af_buffer_info object
/// \returns AF_SUCCESS
AFAPI af_err af_unlock_buffer_info_event(af_event* event, af_buffer_info buf);

/// \brief Disassociates the pointer from the \ref af_buffer_info object
///
/// Gets the pointer and disassociated it from the \ref af_buffer_info object.
/// Deleting the af_buffer_info object will not affect this pointer.
///
/// param[out] event The \ref pointer that will be disassociated. If NULL no
/// pointer is
///                  returned and the data is NOT freed.
/// param[in] buf The target \ref af_buffer_info object
/// \returns AF_SUCCESS
AFAPI af_err af_unlock_buffer_info_ptr(void** ptr, af_buffer_info buf);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // AF_API_VERSION >= 37
