#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <medfilt.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{

template<typename T, af_pad_type pad>
Array<T> * medfilt(const Array<T> &in, dim_type w_len, dim_type w_wid)
{
    Array<T> * out = nullptr;
    OPENCL_NOT_SUPPORTED();
    return out;
}

#define INSTANTIATE(T)\
    template Array<T> * medfilt<T, AF_ZERO     >(const Array<T> &in, dim_type w_len, dim_type w_wid); \
    template Array<T> * medfilt<T, AF_SYMMETRIC>(const Array<T> &in, dim_type w_len, dim_type w_wid);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
