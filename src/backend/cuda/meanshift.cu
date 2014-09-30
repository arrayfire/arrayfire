#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <meanshift.hpp>
#include <err_cuda.hpp>

using af::dim4;

namespace cuda
{

template<typename T, bool is_color>
Array<T> * meanshift(const Array<T> &in, const float &s_sigma, const float &c_sigma, const unsigned iter)
{
    Array<T> *out = 0;
    CUDA_NOT_SUPPORTED();
    return out;
}

#define INSTANTIATE(T) \
    template Array<T> * meanshift<T, true >(const Array<T> &in, const float &s_sigma, const float &c_sigma, const unsigned iter); \
    template Array<T> * meanshift<T, false>(const Array<T> &in, const float &s_sigma, const float &c_sigma, const unsigned iter);

INSTANTIATE(float )
INSTANTIATE(double)
INSTANTIATE(char  )
INSTANTIATE(int   )
INSTANTIATE(uint  )
INSTANTIATE(uchar )

}
