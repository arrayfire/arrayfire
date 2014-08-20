#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <transpose.hpp>

#include <cassert>

using af::dim4;

namespace opencl
{

    template<typename T>
    Array<T> * transpose(const Array<T> &in)
    {
        Array<T> *out = nullptr;
        assert("transpose is not supported yet in opencl backend" && 1<3);
        return out;
    }

#define INSTANTIATE(T)\
    template Array<T> * transpose(const Array<T> &in);

    INSTANTIATE(float)
    INSTANTIATE(cfloat)
    INSTANTIATE(double)
    INSTANTIATE(cdouble)
    INSTANTIATE(char)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)

}
