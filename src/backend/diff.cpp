#include <af/array.h>
#include <af/defines.h>
#include <diff.hpp>
#include <helper.h>
#include <backend.h>

using af::dim4;
using namespace detail;

af_err af_diff1(af_array *out, const af_array in, const int dim)
{
    af_err ret = AF_ERR_RUNTIME;
    af_array output;

    if (dim < 0 || dim > 3) {
        ret = AF_ERR_ARG;
    } else {
        try {
            af_dtype type;
            af_get_type(&type, in);
            switch(type) {
                case f32: output = diff1<float  >(in,dim);       break;
                case c32: output = diff1<cfloat >(in,dim);       break;
                case f64: output = diff1<double >(in,dim);       break;
                case c64: output = diff1<cdouble>(in,dim);       break;
                case b8:  output = diff1<char   >(in,dim);       break;
                case s32: output = diff1<int    >(in,dim);       break;
                case u32: output = diff1<unsigned int >(in,dim); break;
                case u8:  output = diff1<unsigned char>(in,dim); break;
                case s8:  output = diff1<char>(in,dim);          break;
                default:  ret = AF_ERR_NOT_SUPPORTED;            break;
            }
            if (ret!=AF_ERR_NOT_SUPPORTED) {
                std::swap(*out,output);
                ret = AF_SUCCESS;
            }
        }
        CATCHALL;
    }

    return ret;
}
