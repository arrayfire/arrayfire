#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <bilateral.hpp>
#include <stdexcept>

using af::dim4;

namespace opencl
{

template<typename T, bool isColor>
Array<T> * bilateral(const Array<T> &in, const float &s_sigma, const float &c_sigma)
{
    Array<T> *out = nullptr;
    throw std::runtime_error("bilateral not supported in opencl");
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
