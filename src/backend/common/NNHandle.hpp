/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/err_common.hpp>
#include <cstdio>

namespace common {
template <typename T, typename H>
class NNHandle {
   public:
    NNHandle() { static_cast<T *>(this)->createHandle(&handle); }

    ~NNHandle() { static_cast<T *>(this)->destroyHandle(handle); }

    H get() const { return handle; }

   private:
    NNHandle(NNHandle const &);
    void operator=(NNHandle const &);

    H handle;
};
}
