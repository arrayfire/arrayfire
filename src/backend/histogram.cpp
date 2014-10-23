#include <af/dim4.hpp>
#include <af/image.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <backend.hpp>
#include <histogram.hpp>

using af::dim4;
using namespace detail;

template<typename inType,typename outType>
static inline af_array histogram(const af_array in, const unsigned &nbins,
                                 const double &minval, const double &maxval)
{
    return getHandle(*histogram<inType,outType>(getArray<inType>(in),nbins,minval,maxval));
}

af_err af_histogram(af_array *out, const af_array in,
                    const unsigned nbins, const double minval, const double maxval)
{
    try {
        ArrayInfo info = getInfo(in);
        af_dtype type  = info.getType();
        af::dim4 dims  = info.dims();

        DIM_ASSERT(1, (dims.ndims()<=3));

        af_array output;
        switch(type) {
            case f32: output = histogram<float , uint>(in, nbins, minval, maxval); break;
            case f64: output = histogram<double, uint>(in, nbins, minval, maxval); break;
            case b8 : output = histogram<char  , uint>(in, nbins, minval, maxval); break;
            case s32: output = histogram<int   , uint>(in, nbins, minval, maxval); break;
            case u32: output = histogram<uint  , uint>(in, nbins, minval, maxval); break;
            case u8 : output = histogram<uchar , uint>(in, nbins, minval, maxval); break;
            default : TYPE_ERROR(1, type);
        }
        std::swap(*out,output);
    }
    CATCHALL;

    return AF_SUCCESS;
}
