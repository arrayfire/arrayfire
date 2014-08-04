#include <complex>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>
#include <generator.hpp>

using std::swap;
using af::dim4;

namespace cuda {
    template<typename T>
    af_array createArrayHandle(dim4 d, double val)
    {
        return getHandle(*createValueArray<T>(d, val));
    }

    // TODO: See if we can combine specializations
    template<>
    af_array createArrayHandle<cfloat>(dim4 d, double val)
    {
        cfloat cval = {static_cast<float>(val), 0};
        return getHandle(*createValueArray<cfloat>(d, cval));
    }

    template<>
    af_array createArrayHandle<cdouble>(dim4 d, double val)
    {
        cdouble cval = {val, 0};
        return getHandle(*createValueArray<cdouble>(d, cval));
    }

    template af_array createArrayHandle<float>(dim4 d, double val);
    template af_array createArrayHandle<double>(dim4 d, double val);
    template af_array createArrayHandle<int>(dim4 d, double val);
    template af_array createArrayHandle<unsigned>(dim4 d, double val);
    template af_array createArrayHandle<char>(dim4 d, double val);
    template af_array createArrayHandle<unsigned char>(dim4 d, double val);

    template<typename T>
    af_array createArrayHandle(dim4 d, const T * const data)
    {
        return getHandle(*createDataArray<T>(d, data));
    }

    template af_array createArrayHandle<float>(dim4 d, const float * const val);
    template af_array createArrayHandle<double>(dim4 d, const double * const val);
    template af_array createArrayHandle<cfloat>(dim4 d, const cfloat * const val);
    template af_array createArrayHandle<cdouble>(dim4 d, const cdouble * const val);
    template af_array createArrayHandle<int>(dim4 d, const int * const val);
    template af_array createArrayHandle<unsigned>(dim4 d, const unsigned * const val);
    template af_array createArrayHandle<char>(dim4 d, const char * const val);
    template af_array createArrayHandle<unsigned char>(dim4 d, const unsigned char * const val);

    template<typename T>
    void
    destroyArrayHandle(const af_array& arr)
    {
        Array<T> &obj = getWritableArray<T>(arr);
        delete &obj;
    }

    template void destroyArrayHandle<float>                        (const af_array& arr);
    template void destroyArrayHandle<cfloat>                       (const af_array& arr);
    template void destroyArrayHandle<double>                       (const af_array& arr);
    template void destroyArrayHandle<cdouble>                      (const af_array& arr);
    template void destroyArrayHandle<char>                         (const af_array& arr);
    template void destroyArrayHandle<int>                          (const af_array& arr);
    template void destroyArrayHandle<unsigned>                     (const af_array& arr);
    template void destroyArrayHandle<unsigned char>                (const af_array& arr);
}
