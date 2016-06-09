/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/defines.h>
#include <af/moments.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

array moments(const array& in, const af_moment_type moment)
{
    af_array out = 0;
    AF_THROW(af_moments(&out, in.get(), moment));
    return array(out);
}


#define INSTANTIATE_REAL(T)                                 \
    template<> AFAPI                                        \
    T moment(const array &in, const af_moment_type moment)      \
    {                                                       \
        af_array out;                                       \
        AF_THROW(af_moments(&out, in.get(), moment));       \
        return array(out).scalar<T>();                      \
    }                                                       \


INSTANTIATE_REAL(float)
INSTANTIATE_REAL(double)
INSTANTIATE_REAL(int)
INSTANTIATE_REAL(unsigned)
INSTANTIATE_REAL(long long)
INSTANTIATE_REAL(unsigned long long)
INSTANTIATE_REAL(short)
INSTANTIATE_REAL(unsigned short)
INSTANTIATE_REAL(char)
INSTANTIATE_REAL(unsigned char)

#undef INSTANTIATE_REAL

}
