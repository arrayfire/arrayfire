/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/statistics.h>
#include <af/array.h>
#include "error.hpp"

namespace af
{

#define INSTANTIATE_CORRCOEF(T)                                    \
    template<> AFAPI T corrcoef(const array& X, const array& Y)    \
    {                                                              \
        double real;                                               \
        AF_THROW(af_corrcoef(&real, NULL, X.get(), Y.get()));      \
        return (T)real;                                            \
    }                                                              \

INSTANTIATE_CORRCOEF(float);
INSTANTIATE_CORRCOEF(double);
INSTANTIATE_CORRCOEF(int);
INSTANTIATE_CORRCOEF(unsigned int);
INSTANTIATE_CORRCOEF(char);
INSTANTIATE_CORRCOEF(unsigned char);

#undef INSTANTIATE_CORRCOEF

}
