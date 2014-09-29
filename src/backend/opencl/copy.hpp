#include <af/array.h>
#include <Array.hpp>

namespace opencl
{

    template<typename T>
    void copyData(T *data, const Array<T> &A);

    template<typename T>
    Array<T>* copyArray(const Array<T> &A);

    template<typename inType, typename outType>
    void copy(Array<outType> &dst, const Array<inType> &src, double factor);

}
