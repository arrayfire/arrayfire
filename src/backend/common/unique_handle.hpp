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

#include <utility>

namespace arrayfire {
namespace common {

template<typename T>
class ResourceHandler {
   public:
    template<typename... Args>
    static int createHandle(T *handle, Args... args);
    static int destroyHandle(T handle);
};

/// \brief A generic class to manage basic RAII lifetimes for C handles
///
/// This class manages the lifetimes of C handles found in many types of
/// libraries. This class is non-copiable but can be moved.
///
/// You can use this class with a new handle by using the DEFINE_HANDLER
/// macro to define creatHandle/destroyHandle policy implemention for a
/// given resource handle type.
///
/// \code{.cpp}
/// DEFINE_HANDLER(ClassName, HandleName, HandleCreator, HandleDestroyer);
/// \code{.cpp}
template<typename T>
class unique_handle {
   private:
    T handle_;

   public:
    /// Default constructor. Initializes the handle to zero. Does not call the
    /// create function
    constexpr unique_handle() noexcept : handle_(0) {}

    /// \brief Takes ownership of a previously created handle
    ///
    /// \param[in] handle The handle to manage by this object
    explicit constexpr unique_handle(T handle) noexcept : handle_(handle){};

    /// \brief Deletes the handle if created.
    ~unique_handle() noexcept { reset(); }

    /// \brief Deletes the handle if created.
    void reset() noexcept {
        if (handle_) {
            ResourceHandler<T>::destroyHandle(handle_);
            handle_ = 0;
        }
    }

    unique_handle(const unique_handle &other) noexcept      = delete;
    unique_handle &operator=(unique_handle &other) noexcept = delete;

    AF_CONSTEXPR unique_handle(unique_handle &&other) noexcept
        : handle_(other.handle_) {
        other.handle_ = 0;
    }

    unique_handle &operator=(unique_handle &&other) noexcept {
        handle_       = other.handle_;
        other.handle_ = 0;
    }

    /// \brief Implicit converter for the handle
    constexpr operator const T &() const noexcept { return handle_; }

    template<typename... Args>
    int create(Args... args) {
        if (!handle_) {
            int error = ResourceHandler<T>::createHandle(
                &handle_, std::forward<Args>(args)...);
            if (error) { handle_ = 0; }
            return error;
        }
        return 0;
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
///        object is already called with the parameter pack provided as
///        function arguments.
template<typename T, typename... Args>
unique_handle<T> make_handle(Args... args) {
    unique_handle<T> h;
    h.create(std::forward<Args>(args)...);
    return h;
}

}  // namespace common
}  // namespace arrayfire

#define DEFINE_HANDLER(HANDLE_TYPE, HCREATOR, HDESTROYER)            \
    namespace arrayfire {                                            \
    namespace common {                                               \
    template<>                                                       \
    class ResourceHandler<HANDLE_TYPE> {                             \
       public:                                                       \
        template<typename... Args>                                   \
        static int createHandle(HANDLE_TYPE *handle, Args... args) { \
            return HCREATOR(handle, std::forward<Args>(args)...);    \
        }                                                            \
        static int destroyHandle(HANDLE_TYPE handle) {               \
            return HDESTROYER(handle);                               \
        }                                                            \
    };                                                               \
    }                                                                \
    }
