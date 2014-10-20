#include <Array.hpp>
#include <sort_index.hpp>
#include <kernel/sort_index.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_cuda.hpp>

namespace cuda
{
    template<typename T, bool DIR>
    void sort_index(Array<T> &val, Array<unsigned> &idx, const Array<T> &in, const unsigned dim)
    {
        switch(dim) {
            case 0: kernel::sort0_index<T, DIR>(val, idx, in);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(T)                                                                          \
    template void sort_index<T, true>(Array<T> &val, Array<unsigned> &idx, const Array<T> &in,  \
                                      const unsigned dim);                                      \
    template void sort_index<T,false>(Array<T> &val, Array<unsigned> &idx, const Array<T> &in,  \
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
