/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace opencl
{
    template<typename T, typename Tr>
    void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in);

    template<typename T, typename Tr>
    void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in);
}
