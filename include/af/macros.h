/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <stdio.h>

///
/// Print a line on screen using printf syntax.
/// Usage: Uses same syntax and semantics as printf.
/// Output: \<filename\>:\<line number\>: \<message\>
///
#ifndef AF_MSG
#define AF_MSG(fmt,...) do {            \
        printf("%s:%d: " fmt "\n",      \
                 __FILE__, __LINE__, ##__VA_ARGS__);      \
        } while (0);
#endif

