/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <string.h>
#if defined(_WIN32) || defined(_MSC_VER)
    #define __PRETTY_FUNCTION__ __FUNCSIG__
    #if _MSC_VER < 1900
        #define snprintf sprintf_s
    #endif
    #define STATIC_ static
    #define __AF_FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
    //#ifndef __PRETTY_FUNCTION__
    //    #define __PRETTY_FUNCTION__ __func__ // __PRETTY_FUNCTION__ Fallback
    //#endif
    #define STATIC_ inline
    #define __AF_FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

