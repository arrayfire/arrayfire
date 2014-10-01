#include <Array.hpp>
#include <sort.hpp>
#include <kernel/sort.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T>
    void sort(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in, const bool dir, const unsigned dim)
    {
        // TODO Add template for dir
        switch(dim) {
            case 0: kernel::sort0<T>(sx, ix, in, dir);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(T)                                                              \
    template void sort<T>(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in,    \
                          const bool dir, const unsigned dim);                      \

    INSTANTIATE(float)
    INSTANTIATE(double)
    //INSTANTIATE(cfloat)
    //INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
