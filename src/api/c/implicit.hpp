/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <backend.hpp>
#include <common/ArrayInfo.hpp>
#include <common/cast.hpp>
#include <handle.hpp>
#include <optypes.hpp>
#include <types.hpp>
#include <af/array.h>
#include <af/defines.h>

af_dtype implicit(const af_array lhs, const af_array rhs);
af_dtype implicit(const af_dtype lty, const af_dtype rty);
