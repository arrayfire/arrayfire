#include <Array.hpp>
#include <sort.hpp>
#include <kernel/sort.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cuda.hpp>

namespace cuda
{
    template<typename T, bool DIR>
    void sort(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in, const unsigned dim)
    {
        switch(dim) {
            case 0: kernel::sort0<T, DIR>(sx, ix, in);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(T)                                                                        \
    template void sort<T, true>(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in,        \
                                    const unsigned dim);                                      \
    template void sort<T,false>(Array<T> &sx, Array<unsigned> &ix, const Array<T> &in,        \
                                    const unsigned dim);                                      \

    INSTANTIATE(float)
    INSTANTIATE(double)
    //INSTANTIATE(cfloat)
    //INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(char)
    INSTANTIATE(uchar)
}
