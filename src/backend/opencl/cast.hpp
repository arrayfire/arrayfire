#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <complex>
#include <err_opencl.hpp>
#include <math.hpp>

namespace opencl
{
    template<typename To, typename Ti>
    Array<To> *createCastNode(const Array<Ti> &in)
    {
        return createValueArray<To>(in.dims(), scalar<To>(0));
    }

    template<typename To, typename Ti>
    Array<To>* cast(const Array<Ti> &in)
    {
        return createCastNode<To, Ti>(in);
    }

}
