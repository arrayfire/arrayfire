#include <af/array.h>
#include <helper.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>
#include <tile.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array tile(const af_array in, const af::dim4 &tileDims)
{
    return getHandle(*tile<T>(getArray<T>(in), tileDims));
}

af_err af_tile(af_array *out, const af_array in, const af::dim4 &tileDims)
{
    af_err ret = AF_ERR_RUNTIME;

    try {
        ArrayInfo info = getInfo(in);
        af_dtype type = info.getType();

        if(tileDims.elements() == 0)
            return AF_ERR_ARG;
        if(info.dims().elements() == 0)
            return AF_ERR_ARG;

        af_array output;

        switch(type) {
            case f32: output = tile<float  >(in, tileDims);  break;
            case c32: output = tile<cfloat >(in, tileDims);  break;
            case f64: output = tile<double >(in, tileDims);  break;
            case c64: output = tile<cdouble>(in, tileDims);  break;
            case b8:  output = tile<char   >(in, tileDims);  break;
            case s32: output = tile<int    >(in, tileDims);  break;
            case u32: output = tile<uint   >(in, tileDims);  break;
            case u8:  output = tile<uchar  >(in, tileDims);  break;
            case s8:  output = tile<char   >(in, tileDims);  break;
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

af_err af_tile(af_array *out, const af_array in,
               const unsigned x, const unsigned y,
               const unsigned z, const unsigned w)
{
    af::dim4 tileDims(x, y, z, w);
    return af_tile(out, in, tileDims);
}
