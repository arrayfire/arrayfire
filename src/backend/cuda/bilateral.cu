#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <bilateral.hpp>
#include <kernel/bilateral.hpp>
#include <stdexcept>

using af::dim4;

namespace cuda
{

template<typename T, bool isColor>
Array<T> * bilateral(const Array<T> &in, const float &s_sigma, const float &c_sigma)
{
    const dim4 dims     = in.dims();
    const dim4 istrides = in.strides();

    Array<T>* out       = createEmptyArray<T>(dims);
    const dim4 ostrides = out->strides();

    kernel::bilateral_params_t<T> params;
    params.d_dst = out->get();
    params.d_src = in.get();
    for(dim_type i=0; i<4; ++i) {
        params.idims[i]    = dims[i];
        params.istrides[i] = istrides[i];
        params.ostrides[i] = ostrides[i];
    }

    kernel::bilateral<T, isColor>(params, s_sigma, c_sigma);

    return out;
}

#define INSTANTIATE(T)\
template Array<T> * bilateral<T,true >(const Array<T> &in, const float &s_sigma, const float &c_sigma);\
template Array<T> * bilateral<T,false>(const Array<T> &in, const float &s_sigma, const float &c_sigma);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
