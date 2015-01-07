/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/seq.h>
#include <af/array.h>


#ifdef __cplusplus
namespace af
{

AFAPI bool gforToggle();
AFAPI bool gforGet();
AFAPI void gforSet(bool val);


#define gfor(var, ...) for (var = af::seq(af::seq(__VA_ARGS__), true); af::gforToggle(); )

typedef array (*batchFunc_t)(const array &lhs, const array &rhs);
AFAPI array batchFunc(const array &lhs, const array &rhs, batchFunc_t func);

}
#endif
