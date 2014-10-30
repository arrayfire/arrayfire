/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/array.h>
#include <Array.hpp>

namespace cuda
{
    template<typename Tl, typename Tr, typename To, typename Op>
    Array<To>* binOp(const Array<Tl> &lhs, const Array<Tr> &rhs);
}
