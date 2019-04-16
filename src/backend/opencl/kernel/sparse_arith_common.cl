/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

T _add_(T v1, T v2) { return v1 + v2; }

T _sub_(T v1, T v2) { return v1 - v2; }

#if IS_CPLX
T _mul_(T v1, T v2) {
    T out;
    out.x = v1.x * v2.x - v1.y * v2.y;
    out.y = v1.x * v2.y + v1.y * v2.x;
    return out;
}

T _div_(T v1, T v2) {
    T out;
    out.x = (v1.x * v2.x + v1.y * v2.y) / (v2.x * v2.x + v2.y * v2.y);
    out.y = (v1.y * v2.x - v1.x * v2.y) / (v2.x * v2.x + v2.y * v2.y);
    return out;
}
#else
T _mul_(T v1, T v2) { return v1 * v2; }

T _div_(T v1, T v2) { return v1 / v2; }
#endif

#define ADD _add_
#define SUB _sub_
#define MUL _mul_
#define DIV _div_
