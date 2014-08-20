#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/array.h>
#include <transpose.hpp>
#include <helper.h>
#include <backend.h>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array transpose(const af_array in)
{
    return getHandle(*transpose<T>(getArray<T>(in)));
}

af_err af_transpose(af_array *out, af_array in)
{
    af_err ret = AF_ERR_RUNTIME;
    af_array output;

    try {
        af_dtype type;
        af_get_type(&type, in);
        switch(type) {
            case f32: output = transpose<float>(in);          break;
            case c32: output = transpose<cfloat>(in);         break;
            case f64: output = transpose<double>(in);         break;
            case c64: output = transpose<cdouble>(in);        break;
            case b8 : output = transpose<char>(in);           break;
            case s32: output = transpose<int>(in);            break;
            case u32: output = transpose<unsigned>(in);       break;
            case u8 : output = transpose<unsigned char>(in);  break;
            default : ret  = AF_ERR_NOT_SUPPORTED;            break;
        }
        if (ret!=AF_ERR_NOT_SUPPORTED) {
            std::swap(*out,output);
            ret = AF_SUCCESS;
        }
    }
    CATCHALL;

    return ret;
}
