/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/util.h>
#include <type_util.hpp>
#include <err_common.hpp>

const char *getName(af_dtype type)
{
    switch(type) {
    case f32: return "float";
    case f64: return "double";
    case c32: return "complex float";
    case c64: return "complex double";
    case u32: return "unsigned int";
    case s32: return "int";
    case u16: return "unsigned short";
    case s16: return "short";
    case u64: return "unsigned long long";
    case s64: return "long long";
    case u8 : return "unsigned char";
    case b8 : return "bool";
    default : return "unknown type";
    }
}

size_t size_of(af_dtype type)
{
    try {
        switch(type) {
            case f32: return sizeof(float);
            case f64: return sizeof(double);
            case s32: return sizeof(int);
            case u32: return sizeof(unsigned);
            case u8 : return sizeof(unsigned char);
            case b8 : return sizeof(unsigned char);
            case c32: return sizeof(float) * 2;
            case c64: return sizeof(double) * 2;
            case s16: return sizeof(short);
            case u16: return sizeof(unsigned short);
            case s64: return sizeof(intl);
            case u64: return sizeof(uintl);
            default : TYPE_ERROR(1, type);
        }
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_size_of(size_t *size, af_dtype type)
{
    *size = size_of(type);
    return AF_SUCCESS;
}
