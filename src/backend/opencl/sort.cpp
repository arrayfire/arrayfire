#include <Array.hpp>
#include <sort.hpp>
#include <kernel/sort.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
    template<typename T, bool DIR>
    void sort(Array<T> &val, const Array<T> &in, const unsigned dim)
    {
        switch(dim) {
            case 0: kernel::sort0<T, DIR>(val, in);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

    template<typename T, bool DIR>
    void sort_index(Array<T> &val, Array<unsigned> &idx, const Array<T> &in, const unsigned dim)
    {
        switch(dim) {
            case 0: kernel::sort0_index<T, DIR>(val, idx, in);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

    template<typename Tk, typename Tv, bool DIR>
    void sort_by_key(Array<Tk> &okey, Array<Tv> &oval,
               const Array<Tk> &ikey, const Array<Tv> &ival, const unsigned dim)
    {
        switch(dim) {
            case 0: kernel::sort0_by_key<Tk, Tv, DIR>(okey, oval, ikey, ival);
                    break;
            default: AF_ERROR("Not Supported", AF_ERR_NOT_SUPPORTED);
        }
    }

#define INSTANTIATE(T)                                                                          \
    template void sort<T, true>(Array<T> &val, const Array<T> &in, const unsigned dim);         \
    template void sort<T,false>(Array<T> &val, const Array<T> &in, const unsigned dim);         \
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

#define INIT(Tk, Tv)                                                                            \
    template void                                                                               \
    sort_by_key<Tk, Tv, true>(Array<Tk> &okey, Array<Tv> &oval,                                 \
                        const Array<Tk> &ikey, const Array<Tv> &ival, const unsigned dim);      \
    template void                                                                               \
    sort_by_key<Tk, Tv,false>(Array<Tk> &okey, Array<Tv> &oval,                                 \
                        const Array<Tk> &ikey, const Array<Tv> &ival, const unsigned dim);      \

#define INIT1(Tk)       \
    INIT(Tk, float)     \
    INIT(Tk, double)    \
    INIT(Tk, int)       \
    INIT(Tk, uint)      \
    INIT(Tk, char)      \
    INIT(Tk, uchar)     \

    INIT1(float)
    INIT1(double)
    INIT1(int)
    INIT1(uint)
    INIT1(char)
    INIT1(uchar)
}
