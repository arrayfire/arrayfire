#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <morph.hpp>
#include <stdexcept>

using af::dim4;

namespace cuda
{

    template<typename T, bool isDilation>
    Array<T> * morph(const Array<T> &in, const Array<T> &mask)
    {
        const dim4 dims   = in.dims();

        Array<T>* out     = createEmptyArray<T>(dims);

        T* outData        = out->get();
        const T*   inData = in.get();

        throw std::runtime_error("Morphological operations are not supported in cuda yet");

        return out;
    }

    template<typename T, bool isDilation>
    Array<T> * morph3d(const Array<T> &in, const Array<T> &mask)
    {
        const dim4 dims   = in.dims();

        Array<T>* out     = createEmptyArray<T>(dims);

        T* outData        = out->get();
        const T*   inData = in.get();

        throw std::runtime_error("3D Morphological operations are not supported in cuda yet");

        return out;
    }

#define INSTANTIATE(T)\
    template Array<T> * morph  <T, true >(const Array<T> &in, const Array<T> &mask);\
    template Array<T> * morph  <T, false>(const Array<T> &in, const Array<T> &mask);\
    template Array<T> * morph3d<T, true >(const Array<T> &in, const Array<T> &mask);\
    template Array<T> * morph3d<T, false>(const Array<T> &in, const Array<T> &mask);

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(char)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)

}
