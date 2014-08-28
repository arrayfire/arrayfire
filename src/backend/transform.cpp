#include <af/image.h>
#include <af/defines.h>
#include <transform.hpp>
#include <helper.hpp>
#include <backend.hpp>
#include <ArrayInfo.hpp>

using af::dim4;
using namespace detail;

template<typename T>
static inline af_array transform(const af_array in, const af_array tf, const af::dim4 &odims, const bool inverse)
{
    return getHandle(*transform<T>(getArray<T>(in), getArray<float>(tf), odims, inverse));
}

af_err af_transform(af_array *out, const af_array in, const af_array tf,
                    const dim_type odim0, const dim_type odim1, const bool inverse)
{
    af_err ret = AF_SUCCESS;
    try {
        //if(tf is not float)
        //    return AF_ERR_ARG;

        ArrayInfo t_info = getInfo(tf);
        ArrayInfo i_info = getInfo(in);

        af::dim4 idims = i_info.dims();
        af::dim4 tdims = t_info.dims();
        af_dtype  type = i_info.getType();

        if(tdims[0] != 3 || tdims[1] != 2)
            return AF_ERR_ARG;

        if(idims.elements() == 0)
            return AF_ERR_ARG;

        if(idims.ndims() < 2 || idims.ndims() > 3)
            return AF_ERR_ARG;

        dim_type o0 = odim0, o1 = odim1;
        dim_type o2 = idims[2] * tdims[2];
        if (odim0 * odim1 == 0) {
            o0 = idims[0];
            o1 = idims[1];
        }
        af::dim4 odims(o0, o1, o2, 1);

        af_array output = 0;
        switch(type) {
            case f32: output = transform<float  >(in, tf, odims, inverse);  break;
            case f64: output = transform<double >(in, tf, odims, inverse);  break;
            case s32: output = transform<int    >(in, tf, odims, inverse);  break;
            case u32: output = transform<uint   >(in, tf, odims, inverse);  break;
            case u8:  output = transform<uchar  >(in, tf, odims, inverse);  break;
          //case c32: output = transform<cfloat >(in, tf, odims, inverse);  break;
          //case c64: output = transform<cdouble>(in, tf, odims, inverse);  break;
          //case b8:  output = transform<char   >(in, tf, odims, inverse);  break;
          //case s8:  output = transform<char   >(in, tf, odims, inverse);  break;
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

af_err af_rotate(af_array *out, const af_array in, const float theta,
                 const bool crop, const bool recenter)
{
    af_err ret = AF_SUCCESS;
    try {
        unsigned odims0 = 0, odims1 = 0;
        float c = std::cos(-theta), s = std::sin(-theta);
        float tx = 0, ty = 0;

        ArrayInfo info = getInfo(in);
        af::dim4 idims = info.dims();

        if(!crop) {
            odims0 = idims[0] * fabs(std::cos(theta)) + idims[1] * fabs(std::sin(theta));
            odims1 = idims[1] * fabs(std::cos(theta)) + idims[0] * fabs(std::sin(theta));
        } else {
            odims0 = idims[0];
            odims1 = idims[1];
        }

        if (recenter) { //Find new coordintates of center and translate it
            float nx = 0.5 * (idims[0] - 1);
            float ny = 0.5 * (idims[1] - 1);
            float mx = 0.5 * (odims0 - 1);
            float my = 0.5 * (odims1 - 1);
            float sx = (mx * c + my *-s);
            float sy = (mx * s + my * c);
            tx = -(sx - nx);
            ty = -(sy - ny);
        }

        //Correct transform matrix for forward rotation
        static float trans_mat[6] = {1, 0, 0,
                                     0, 1, 0};
        trans_mat[0] =  c;
        trans_mat[1] = -s;
        trans_mat[2] = tx;
        trans_mat[3] =  s;
        trans_mat[4] =  c;
        trans_mat[5] = ty;

        //If inverse, generated inverse matrix
        //if(inverse)
        //    calc_transform_inverse(trans_mat);

        af::dim4 tdims(3, 2, 1, 1);
        af_array t = 0;
        ret = af_create_array(&t, trans_mat, tdims.ndims(), tdims.get(), f32);

        if (ret == AF_SUCCESS) {
            return af_transform(out, in, t, odims0, odims1, true);
        }
    }
    CATCHALL;

    return ret;
}
