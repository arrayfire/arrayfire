/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <err_opencl.hpp>
#include <stdio.h>
#include <errorcodes.hpp>

#ifndef NDEBUG
#include <iostream>
#define CL_DEBUG_FINISH(Q) Q.finish()
#else
#define CL_DEBUG_FINISH(Q)
#endif
