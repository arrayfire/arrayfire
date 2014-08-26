#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <histogram.hpp>
#include <cassert>

using af::dim4;

namespace opencl
{

template<typename inType, typename outType>
Array<outType> * histogram(const Array<inType> &in, const unsigned &nbins, const double &minval, const double &maxval)
{
    Array<outType>* out  = nullptr;
    assert("histogram not implemented yet in opencl backend" && out!=nullptr);
    return out;
}

#define INSTANTIATE(in_t,out_t)\
template Array<out_t> * histogram(const Array<in_t> &in, const unsigned &nbins, const double &minval, const double &maxval);

INSTANTIATE(float,uint)
INSTANTIATE(double,uint)
INSTANTIATE(char,uint)
INSTANTIATE(int,uint)
INSTANTIATE(uint,uint)
INSTANTIATE(uchar,uint)

}
