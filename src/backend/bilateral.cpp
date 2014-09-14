#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/image.h>
#include <handle.hpp>
#include <backend.hpp>
#include <bilateral.hpp>
#include <err_common.hpp>

using af::dim4;
using namespace detail;

template<typename T, bool isColor>
static inline af_array bilateral(const af_array &in, const float &sp_sig, const float &chr_sig)
{
    return getHandle(*bilateral<T, isColor>(getArray<T>(in), sp_sig, chr_sig));
}

template<bool isColor>
static af_err bilateral(af_array *out, const af_array &in, const float &s_sigma, const float &c_sigma)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 dims  = info.dims();

        if (isColor) {
            DIM_ASSERT(1, (dims.ndims()>=3));
        } else {
            DIM_ASSERT(1, (dims.ndims()>=2 && dims.ndims()<=3));
        }

        af_array output;
        switch(type) {
            case f32: output = bilateral<float  , isColor> (in, s_sigma, c_sigma); break;
            case f64: output = bilateral<double , isColor> (in, s_sigma, c_sigma); break;
            case b8 : output = bilateral<char   , isColor> (in, s_sigma, c_sigma); break;
            case s32: output = bilateral<int    , isColor> (in, s_sigma, c_sigma); break;
            case u32: output = bilateral<uint   , isColor> (in, s_sigma, c_sigma); break;
            case u8 : output = bilateral<uchar  , isColor> (in, s_sigma, c_sigma); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_bilateral(af_array *out, const af_array in, const float spatial_sigma, const float chromatic_sigma, const bool isColor)
{
    if (isColor)
        return bilateral<true>(out,in,spatial_sigma,chromatic_sigma);
    else
        return bilateral<false>(out,in,spatial_sigma,chromatic_sigma);
}
