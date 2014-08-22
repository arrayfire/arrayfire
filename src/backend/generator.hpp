#include <af/array.h>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <copy.hpp>
#include <backend.hpp>

template<typename T>
static af_array createHandle(af::dim4 d, double val)
{
    return detail::getHandle(*detail::createValueArray<T>(d, static_cast<T>(val)));
}

// TODO: See if we can combine specializations
template<>
af_array createHandle<detail::cfloat>(af::dim4 d, double val)
{
    detail::cfloat cval = {static_cast<float>(val), 0};
    return detail::getHandle(*detail::createValueArray<detail::cfloat>(d, cval));
}

template<>
af_array createHandle<detail::cdouble>(af::dim4 d, double val)
{
    detail::cdouble cval = {val, 0};
    return detail::getHandle(*detail::createValueArray<detail::cdouble>(d, cval));
}

template<typename T>
static af_array createHandle(af::dim4 d, const T * const data)
{
    return detail::getHandle(*detail::createDataArray<T>(d, data));
}

template<typename T>
static void copyData(T *data, const af_array &arr)
{
    return detail::copyData(data, detail::getArray<T>(arr));
}
