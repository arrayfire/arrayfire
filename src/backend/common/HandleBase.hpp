/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

namespace arrayfire {
namespace common {
template<typename T, typename H>
class HandleBase {
    H handle_;

   public:
    HandleBase() : handle_(0) { static_cast<T*>(this)->createHandle(&handle_); }
    ~HandleBase() { static_cast<T*>(this)->destroyHandle(handle_); }

    operator H() { return handle_; }
    H* get() { return &handle_; }

    HandleBase(HandleBase const&)     = delete;
    void operator=(HandleBase const&) = delete;

    HandleBase(HandleBase&& h)            = default;
    HandleBase& operator=(HandleBase&& h) = default;
};
}  // namespace common
}  // namespace arrayfire

#define CREATE_HANDLE(NAME, TYPE, CREATE_FUNCTION, DESTROY_FUNCTION,  \
                      CHECK_FUNCTION)                                 \
    class NAME : public common::HandleBase<NAME, TYPE> {              \
       public:                                                        \
        void createHandle(TYPE* handle) {                             \
            CHECK_FUNCTION(CREATE_FUNCTION(handle));                  \
        }                                                             \
        void destroyHandle(TYPE handle) { DESTROY_FUNCTION(handle); } \
    };
