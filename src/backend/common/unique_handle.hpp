/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/compilers.h>

namespace common {

/// Deletes a handle.
///
/// This function deletes a handle. Handle are usually typedefed pointers
/// which are created by a C API of a library.
///
/// \param[in] handle the handle that will deleted by the destroy function
/// \note This function will need to be specialized for each type of handle
template<typename T>
void handle_deleter(T handle) noexcept;

/// Creates a handle
/// This function creates a handle. Handle are usually typedefed pointers
/// which are created by a C API of a library.
///
/// \param[in] handle the handle that will be initialzed by the create function
/// \note This function will need to be specialized for each type of handle
template<typename T>
int handle_creator(T *handle) noexcept;

/// \brief A generic class to manage basic RAII lifetimes for C handles
///
/// This class manages the lifetimes of C handles found in many types of
/// libraries. This class is non-copiable but can be moved.
///
/// You can use this class with a new handle by using the CREATE_HANDLE macro in
/// the src/backend/*/handle.cpp file. This macro instantiates the
/// handle_createor and handle_deleter functions used by this class.
///
/// \code{.cpp}
/// CREATE_HANDLE(cusparseHandle_t, cusparseCreate, cusparseDestroy);
/// \code{.cpp}
template<typename T>
class unique_handle {
    T handle_;

   public:
    /// Default constructor. Initializes the handle to zero. Does not call the
    /// create function
    constexpr unique_handle() noexcept : handle_(0) {}
    int create() {
        if (!handle_) {
            int error = handle_creator(&handle_);
            if (error) { handle_ = 0; }
            return error;
        }
        return 0;
    }

    /// \brief Takes ownership of a previously created handle
    ///
    /// \param[in] handle The handle to manage by this object
    explicit constexpr unique_handle(T handle) noexcept : handle_(handle){};

    /// \brief Deletes the handle if created.
    ~unique_handle() noexcept {
        if (handle_) handle_deleter(handle_);
    };

    /// \brief Implicit converter for the handle
    constexpr operator const T &() const noexcept { return handle_; }

    unique_handle(const unique_handle &other) noexcept = delete;
    AF_CONSTEXPR unique_handle(unique_handle &&other) noexcept
        : handle_(other.handle_) {
        other.handle_ = 0;
    }

    unique_handle &operator=(unique_handle &other) noexcept = delete;
    unique_handle &operator=(unique_handle &&other) noexcept {
        handle_       = other.handle_;
        other.handle_ = 0;
    }

    // Returns true if the \p other unique_handle is the same as this handle
    constexpr bool operator==(unique_handle &other) const noexcept {
        return handle_ == other.handle_;
    }

    // Returns true if the \p other handle is the same as this handle
    constexpr bool operator==(T &other) const noexcept {
        return handle_ == other;
    }

    // Returns true if the \p other handle is the same as this handle
    constexpr bool operator==(T other) const noexcept {
        return handle_ == other;
    }

    // Returns true if the handle was initialized correctly
    constexpr operator bool() { return handle_ != 0; }
};

/// \brief Returns an initialized handle object. The create function on this
///        object is already called
template<typename T>
unique_handle<T> make_handle() {
    unique_handle<T> h;
    h.create();
    return h;
}

}  // namespace common

/// specializes the handle_creater and handle_deleter functions for a specific
/// handle
///
/// \param[in] HANDLE The type of the handle
/// \param[in] CREATE The create function for the handle
/// \param[in] DESTROY The destroy function for the handle
/// \note Do not add this macro to another namespace, The macro provides a
///       namespace for the functions.
#define CREATE_HANDLE(HANDLE, CREATE, DESTROY)             \
    namespace common {                                     \
    template<>                                             \
    void handle_deleter<HANDLE>(HANDLE handle) noexcept {  \
        DESTROY(handle);                                   \
    }                                                      \
    template<>                                             \
    int handle_creator<HANDLE>(HANDLE * handle) noexcept { \
        return CREATE(handle);                             \
    }                                                      \
    }  // namespace common
