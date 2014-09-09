#include <af/array.h>
#include <af/defines.h>
#include <helper.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <resize.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array resize(const af_array in, const dim_type odim0, const dim_type odim1, const af_interp_type method)
{
    return getHandle(*resize<T>(getArray<T>(in), odim0, odim1, method));
}

af_err af_resize(af_array *out, const af_array in, const dim_type odim0, const dim_type odim1, const af_interp_type method)
{
    af_err ret = AF_SUCCESS;
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();
        af::dim4 dims = info.dims();

        if(dims.ndims() < 2 || dims.ndims() > 3)
            return AF_ERR_ARG;

        if(method != AF_INTERP_NEAREST && method != AF_INTERP_BILINEAR)
            return AF_ERR_ARG;

        if(odim0 <= 0 || odim1 <= 0)
            return AF_ERR_ARG;

        af_array output;

        switch(type) {
            case f32: output = resize<float  >(in, odim0, odim1, method);  break;
            case f64: output = resize<double >(in, odim0, odim1, method);  break;
            case b8:  output = resize<char   >(in, odim0, odim1, method);  break;
            case s32: output = resize<int    >(in, odim0, odim1, method);  break;
            case u32: output = resize<uint   >(in, odim0, odim1, method);  break;
            case u8:  output = resize<uchar  >(in, odim0, odim1, method);  break;
            case s8:  output = resize<char   >(in, odim0, odim1, method);  break;
            default:  ret = AF_ERR_NOT_SUPPORTED;                           break;
        }
        if (ret != AF_ERR_NOT_SUPPORTED || ret != AF_ERR_ARG) {
            std::swap(*out,output);
            ret = AF_SUCCESS;
        }
    }
    CATCHALL;

    return ret;
}
