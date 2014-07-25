#include <complex>
#include <af/dim4.hpp>
#include <af/defines.h>
#include <af/generator.h>
#include <ArrayInfo.hpp>
#include <Array.hpp>

using std::swap;
using af::dim4;
namespace cpu {
    template<typename T>
    af_array createArrayHandle(dim4 d, double val)
    {
        return getHandle(*createArray<T>(d, static_cast<T>(val)));
    }

    // TODO: See if we can combine specializations
    template<>
    af_array createArrayHandle<cfloat>(dim4 d, double val)
    {
        cfloat cval = {static_cast<float>(val), 0};
        return getHandle(*createArray<cfloat>(d, cval));
    }

    template<>
    af_array createArrayHandle<cdouble>(dim4 d, double val)
    {
        cdouble cval = {val, 0};
        return getHandle(*createArray<cdouble>(d, cval));
    }

    template af_array createArrayHandle<float>(dim4 d, double val);
    template af_array createArrayHandle<double>(dim4 d, double val);
    template af_array createArrayHandle<int>(dim4 d, double val);
    template af_array createArrayHandle<unsigned>(dim4 d, double val);
    template af_array createArrayHandle<char>(dim4 d, double val);
    template af_array createArrayHandle<unsigned char>(dim4 d, double val);
}
