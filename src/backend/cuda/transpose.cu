#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <transpose.hpp>

#include <cassert>

using af::dim4;

namespace cuda {

    template<typename T>
    Array<T> * transpose(const Array<T> &in)
    {
        Array<T> *out;
        assert("transpose is not supported yet in cuda backend" && 1==0);
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
    INSTANTIATE(unsigned)
    INSTANTIATE(unsigned char)

}
