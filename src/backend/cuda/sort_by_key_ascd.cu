#include <sort_by_key_impl.hpp>

namespace cuda
{
#define INSTANTIATE(Tk, Tv)                                             \
    template void                                                       \
    sort_by_key<Tk, Tv, true>(Array<Tk> &okey, Array<Tv> &oval,         \
                              const Array<Tk> &ikey, const Array<Tv> &ival, const uint dim); \

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
