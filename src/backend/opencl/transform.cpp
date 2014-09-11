#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <transform.hpp>
#include <kernel/transform.hpp>
#include <stdexcept>

namespace opencl
{
    template<typename T>
    Array<T>* transform(const Array<T> &in, const Array<float> &transform, const af::dim4 &odims,
                        const bool inverse)
    {
        Array<T> *out = createEmptyArray<T>(odims);

        if (inverse) {
            kernel::transform<T, true> (*out, in, transform);
        } else {
            kernel::transform<T, false>(*out, in, transform);
        }

        return out;
    }


#define INSTANTIATE(T)                                                                          \
    template Array<T>* transform(const Array<T> &in, const Array<float> &transform,             \
                                 const af::dim4 &odims, const bool inverse);                    \


    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
}

