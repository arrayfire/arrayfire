/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/err_common.hpp>
#include <cstdio>

namespace common
{
template<typename T, typename H>
class MatrixAlgebraHandle
{
    public:
        MatrixAlgebraHandle() {
            static_cast<T*>(this)->createHandle(&handle);
        }

        ~MatrixAlgebraHandle() {
            static_cast<T*>(this)->destroyHandle(handle);
        }

        H get() const {
            return handle;
        }

    private:
        MatrixAlgebraHandle(MatrixAlgebraHandle const&);
        void operator=(MatrixAlgebraHandle const&);

        H handle;
};
}
