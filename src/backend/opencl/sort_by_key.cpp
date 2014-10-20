#include <Array.hpp>
#include <sort_by_key.hpp>
#include <kernel/sort_by_key.hpp>
#include <math.hpp>
#include <stdexcept>
#include <err_opencl.hpp>

namespace opencl
{
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

#define INSTANTIATE(Tk, Tv)                                                                     \
    template void                                                                               \
    sort_by_key<Tk, Tv, true>(Array<Tk> &okey, Array<Tv> &oval,                                 \
                        const Array<Tk> &ikey, const Array<Tv> &ival, const unsigned dim);      \
    template void                                                                               \
    sort_by_key<Tk, Tv,false>(Array<Tk> &okey, Array<Tv> &oval,                                 \
                        const Array<Tk> &ikey, const Array<Tv> &ival, const unsigned dim);      \

#define INSTANTIATE1(Tk)       \
    INSTANTIATE(Tk, float)     \
    INSTANTIATE(Tk, double)    \
    INSTANTIATE(Tk, int)       \
    INSTANTIATE(Tk, uint)      \
    INSTANTIATE(Tk, char)      \
    INSTANTIATE(Tk, uchar)     \

    INSTANTIATE1(float)
    INSTANTIATE1(double)
    INSTANTIATE1(int)
    INSTANTIATE1(uint)
    INSTANTIATE1(char)
    INSTANTIATE1(uchar)
}
