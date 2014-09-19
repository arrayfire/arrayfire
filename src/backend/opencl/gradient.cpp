#include <Array.hpp>
#include <gradient.hpp>
#include <math.hpp>
#include <kernel/gradient.hpp>
#include <stdexcept>

namespace opencl
{
    template<typename T>
    void gradient(Array<T> &grad0, Array<T> &grad1, const Array<T> &in)
    {
        kernel::gradient<T>(grad0, grad1, in);
    }

#define INSTANTIATE(T)                                                                  \
    template void gradient<T>(Array<T> &grad0, Array<T> &grad1, const Array<T> &in);    \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
}

