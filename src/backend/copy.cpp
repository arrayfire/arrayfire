#include <iostream>

#include <af/array.h>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <copy.hpp>
#include <common_helper.h>
#include <backend_helper.h>
//#include <Array.hpp>

af_err af_copy(af_array *dst, const void* const src)
{
    af_err ret = AF_ERR_RUNTIME;

    try {
        if(dst) {
            const ArrayInfo info = getInfo(*dst);
            switch(info.getType()) {
                case f32:   copyData<float>(*dst, static_cast<const float* const>(src));    break;
                case c32:   copyData<cfloat>(*dst, static_cast<const cfloat* const>(src));    break;
                case f64:   copyData<double>(*dst, static_cast<const double* const>(src));    break;
                case c64:   copyData<cdouble>(*dst, static_cast<const cdouble* const>(src));    break;
                case b8:    copyData<char>(*dst, static_cast<const char* const>(src));    break;
                case s32:   copyData<int>(*dst, static_cast<const int* const>(src));    break;
                case u32:   copyData<unsigned>(*dst, static_cast<const unsigned* const>(src));    break;
                case u8:    copyData<unsigned char>(*dst, static_cast<const unsigned char* const>(src));    break;
                case s8:    copyData<char>(*dst, static_cast<const char* const>(src));    break;
                default:    break;
            }
            ret = AF_SUCCESS;
        }
        else {
            ret = AF_ERR_INVALID_ARRAY;
        }
    }
    CATCHALL

    return ret;
}

af_err af_host_ptr(void **ptr, af_array arr)
{
    af_err ret = AF_SUCCESS;

    try {
        switch(af_get_type(arr)) {
            case f32:   *ptr = copyData<float>(arr);                   break;
            case c32:   *ptr = copyData<cfloat>(arr);                  break;
            case f64:   *ptr = copyData<double>(arr);                  break;
            case c64:   *ptr = copyData<cdouble>(arr);                 break;
            case b8:    *ptr = copyData<char>(arr);                    break;
            case s32:   *ptr = copyData<int>(arr);                     break;
            case u32:   *ptr = copyData<unsigned>(arr);                break;
            case u8:    *ptr = copyData<unsigned char>(arr);           break;
            case s8:    *ptr = copyData<char>(arr);                    break;
            default:    ret  = AF_ERR_RUNTIME;                         break;
        }
    }
    CATCHALL

    return ret;
}
