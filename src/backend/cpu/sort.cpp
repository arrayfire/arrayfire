#include <Array.hpp>
#include <sort.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cpu.hpp>

namespace cpu
{
    template<typename T>
    void sort(Array<T> &sx, Array<T> &ix, const Array<T> &in, const bool dir, const unsigned dim)
    {
        AF_ERROR("Implementation not complete", AF_ERR_NOT_SUPPORTED);
    }

#define INSTANTIATE(T)                                                                                          \
    template void sort<T>(Array<T> &sx, Array<T> &ix, const Array<T> &in, const bool dir, const unsigned dim);  \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
