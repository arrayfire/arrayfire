#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/image.h>
#include <handle.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <medfilt.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static af_array medfilt(af_array const &in, dim_type w_len, dim_type w_wid, af_pad_type edge_pad)
{
    switch(edge_pad) {
        case AF_ZERO     : return getHandle<T>(*medfilt<T, AF_ZERO     >(getArray<T>(in), w_len, w_wid)); break;
        case AF_SYMMETRIC: return getHandle<T>(*medfilt<T, AF_SYMMETRIC>(getArray<T>(in), w_len, w_wid)); break;
        default          : return getHandle<T>(*medfilt<T, AF_ZERO     >(getArray<T>(in), w_len, w_wid)); break;
    }
}

af_err af_medfilt(af_array *out, const af_array in, dim_type wind_length, dim_type wind_width, af_pad_type edge_pad)
{
    try {
        ARG_ASSERT(2, (wind_length==wind_width));
        ARG_ASSERT(2, (wind_length>0));
        ARG_ASSERT(3, (wind_width>0));
        ARG_ASSERT(4, (edge_pad>=AF_ZERO && edge_pad<=AF_SYMMETRIC));

        ArrayInfo info = getInfo(in);
        af::dim4 dims  = info.dims();

        dim_type in_ndims = dims.ndims();
        DIM_ASSERT(1, (in_ndims <= 3 && in_ndims >= 2));

        if (wind_length==1) {
            *out = weakCopy(in);
        } else {
            af_array output;
            af_dtype type  = info.getType();
            switch(type) {
                case f32: output = medfilt<float >(in, wind_length, wind_width, edge_pad); break;
                case f64: output = medfilt<double>(in, wind_length, wind_width, edge_pad); break;
                case b8 : output = medfilt<char  >(in, wind_length, wind_width, edge_pad); break;
                case s32: output = medfilt<int   >(in, wind_length, wind_width, edge_pad); break;
                case u32: output = medfilt<uint  >(in, wind_length, wind_width, edge_pad); break;
                case u8 : output = medfilt<uchar >(in, wind_length, wind_width, edge_pad); break;
                default : TYPE_ERROR(1, type);
            }
            std::swap(*out, output);
        }
    }
    CATCHALL;

    return AF_SUCCESS;
}
