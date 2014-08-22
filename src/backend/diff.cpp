#include <af/array.h>
#include <af/defines.h>
#include <diff.hpp>
#include <helper.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <diff.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array diff1(const af_array in, const int dim)
{
    return getHandle(*diff1<T>(getArray<T>(in), dim));
}

template<typename T>
static inline af_array diff2(const af_array in, const int dim)
{
    return getHandle(*diff2<T>(getArray<T>(in), dim));
}

af_err af_diff1(af_array *out, const af_array in, const int dim)
{
    if (dim < 0 || dim > 3) {
        return AF_ERR_ARG;
    }

    af_err ret = AF_ERR_RUNTIME;

    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        af::dim4 dims = info.dims();

        if(dims[dim] < 2) {
            return AF_ERR_ARG;
        }

        af_array output;

        switch(type) {
            case f32: output = diff1<float  >(in,dim);  break;
            case c32: output = diff1<cfloat >(in,dim);  break;
            case f64: output = diff1<double >(in,dim);  break;
            case c64: output = diff1<cdouble>(in,dim);  break;
            case b8:  output = diff1<char   >(in,dim);  break;
            case s32: output = diff1<int    >(in,dim);  break;
            case u32: output = diff1<uint   >(in,dim);  break;
            case u8:  output = diff1<uchar  >(in,dim);  break;
            case s8:  output = diff1<char   >(in,dim);  break;
            default:  ret = AF_ERR_NOT_SUPPORTED;       break;
        }
        if (ret!=AF_ERR_NOT_SUPPORTED) {
            std::swap(*out,output);
            ret = AF_SUCCESS;
        }
    }
    CATCHALL;

    return ret;
}

af_err af_diff2(af_array *out, const af_array in, const int dim)
{
    if (dim < 0 || dim > 3) {
        return AF_ERR_ARG;
    }

    af_err ret = AF_ERR_RUNTIME;

    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        af::dim4 dims = info.dims();

        if(dims[dim] < 3) {
            return AF_ERR_ARG;
        }

        af_array output;

        switch(type) {
            case f32: output = diff2<float  >(in,dim);  break;
            case c32: output = diff2<cfloat >(in,dim);  break;
            case f64: output = diff2<double >(in,dim);  break;
            case c64: output = diff2<cdouble>(in,dim);  break;
            case b8:  output = diff2<char   >(in,dim);  break;
            case s32: output = diff2<int    >(in,dim);  break;
            case u32: output = diff2<uint   >(in,dim);  break;
            case u8:  output = diff2<uchar  >(in,dim);  break;
            case s8:  output = diff2<char   >(in,dim);  break;
            default:  ret = AF_ERR_NOT_SUPPORTED;       break;
        }
        if (ret!=AF_ERR_NOT_SUPPORTED) {
            std::swap(*out,output);
            ret = AF_SUCCESS;
        }
    }
    CATCHALL;

    return ret;
}
