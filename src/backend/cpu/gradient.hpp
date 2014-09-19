#include <af/array.h>
#include <Array.hpp>

namespace cpu
{
    template<typename T>
    void gradient(Array<T> &grad0, Array<T> &grad1, const Array<T> &in);
}
