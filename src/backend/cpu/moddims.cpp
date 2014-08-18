#include <type_traits>
#include <af/array.h>
#include <Array.hpp>
#include <moddims.hpp>

namespace cpu
{
    template<typename T>
    af_array moddims(const af_array &arr, const af::dim4 &newDims)
    {
        const Array<T> &inArray = getArray<T>(arr);
        Array<T> *out = copyArray(inArray);
        out->eval();
        out->moddims(newDims);
        return getHandle(*out);
    }

    template af_array moddims<float>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<cfloat>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<double>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<cdouble>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<char>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<int>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<unsigned>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<unsigned char>(const af_array &arr, const af::dim4 &newDims);
}
