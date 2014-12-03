/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel
void
set(    global  T*      ptr,
                T       val,
        const   unsigned long  elements)
{
    if(get_global_id(0) < elements) {
        ptr[get_global_id(0)] = val;
    }
}

