/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if defined(_WIN32) || defined(_MSC_VER)
    #define __PRETTY_FUNCTION__ __FUNCSIG__
    #if _MSC_VER < 1900
        #define snprintf sprintf_s
    #endif
    #define STATIC_ static
#else
    #define __PRETTY_FUNCTION__ __func__
    #define STATIC_ inline
#endif
