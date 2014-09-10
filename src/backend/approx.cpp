#include <af/array.h>
#include <af/defines.h>
#include <err_common.hpp>
#include <handle.hpp>
#include <ArrayInfo.hpp>
#include <backend.hpp>
#include <approx.hpp>

using af::dim4;
using namespace detail;

template<typename Ty, typename Tp>
static inline af_array approx1(const af_array in, const af_array pos,
                               const af_interp_type method, const float offGrid)
{
    return getHandle(*approx1<Ty>(getArray<Ty>(in), getArray<Tp>(pos), method, offGrid));
}

template<typename Ty, typename Tp>
static inline af_array approx2(const af_array in, const af_array pos0, const af_array pos1,
                               const af_interp_type method, const float offGrid)
{
    return getHandle(*approx2<Ty>(getArray<Ty>(in), getArray<Tp>(pos0), getArray<Tp>(pos1),
                                 method, offGrid));
}

af_err af_approx1(af_array *out, const af_array in, const af_array pos,
                  const af_interp_type method, const float offGrid)
{
    af_err ret = AF_SUCCESS;
    try {
        ArrayInfo i_info = getInfo(in);
        ArrayInfo p_info = getInfo(pos);

        af_dtype itype = i_info.getType();

        if (!i_info.isFloating())                       // Only floating and complex types
            return AF_ERR_ARG;
        if (!p_info.isRealFloating())                   // Only floating types
            return AF_ERR_ARG;
        if (i_info.isSingle() ^ p_info.isSingle())      // Must have same precision
            return AF_ERR_ARG;
        if (i_info.isDouble() ^ p_info.isDouble())      // Must have same precision
            return AF_ERR_ARG;
        if (!p_info.isColumn())                         // Only 1D input allowed
            return AF_ERR_ARG;
        if (method != AF_INTERP_LINEAR && method != AF_INTERP_NEAREST)
            return AF_ERR_ARG;

        af_array output;

        switch(itype) {
            case f32: output = approx1<float  , float >(in, pos, method, offGrid);  break;
            case f64: output = approx1<double , double>(in, pos, method, offGrid);  break;
            case c32: output = approx1<cfloat , float >(in, pos, method, offGrid);  break;
            case c64: output = approx1<cdouble, double>(in, pos, method, offGrid);  break;
            default:  ret = AF_ERR_NOT_SUPPORTED;                                   break;
        }
        if (ret != AF_ERR_NOT_SUPPORTED || ret != AF_ERR_ARG) {
            std::swap(*out,output);
            ret = AF_SUCCESS;
        }
    }
    CATCHALL;

    return ret;
}

af_err af_approx2(af_array *out, const af_array in, const af_array pos0, const af_array pos1,
                  const af_interp_type method, const float offGrid)
{
    af_err ret = AF_SUCCESS;
    try {
        ArrayInfo i_info = getInfo(in);
        ArrayInfo p_info = getInfo(pos0);
        ArrayInfo q_info = getInfo(pos1);

        af_dtype itype = i_info.getType();

        if (!i_info.isFloating())                       // Only floating and complex types
            return AF_ERR_ARG;
        if (!p_info.isRealFloating())                   // Only floating types
            return AF_ERR_ARG;
        if (!q_info.isRealFloating())                   // Only floating types
            return AF_ERR_ARG;
        if (p_info.getType() != q_info.getType())       // Must be same types
            return AF_ERR_ARG;
        if (i_info.isSingle() ^ p_info.isSingle())      // Must have same precision
            return AF_ERR_ARG;
        if (i_info.isDouble() ^ p_info.isDouble())      // Must have same precision
            return AF_ERR_ARG;
        if (p_info.dims() != q_info.dims())             // POS0 and POS1 must have same dimensions
            return AF_ERR_ARG;
        if (method != AF_INTERP_LINEAR && method != AF_INTERP_NEAREST)
            return AF_ERR_ARG;
        if (p_info.ndims() > 2) // Allowing input batch but not positions. Output will be (px, py, iz, iw)
            return AF_ERR_ARG;

        af_array output;

        switch(itype) {
            case f32: output = approx2<float  , float >(in, pos0, pos1, method, offGrid);  break;
            case f64: output = approx2<double , double>(in, pos0, pos1, method, offGrid);  break;
            case c32: output = approx2<cfloat , float >(in, pos0, pos1, method, offGrid);  break;
            case c64: output = approx2<cdouble, double>(in, pos0, pos1, method, offGrid);  break;
            default:  ret = AF_ERR_NOT_SUPPORTED;                                          break;
        }
        if (ret != AF_ERR_NOT_SUPPORTED || ret != AF_ERR_ARG) {
            std::swap(*out,output);
            ret = AF_SUCCESS;
        }
    }
    CATCHALL;

    return ret;
}
