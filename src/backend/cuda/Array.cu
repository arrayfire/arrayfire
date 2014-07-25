#include <cassert>
#include <af/dim4.hpp>
#include <Array.hpp>
#include <iostream>


namespace cuda
{
    using std::ostream;

    template<typename T>
    const Array<T> &
    getArray(const af_array &arr)
    {
        Array<T> *out = reinterpret_cast<Array<T>*>(arr);
        return *out;
    }

    template const Array<float>&                            getArray<float>(const af_array &arr);
    template const Array<cfloat>&                           getArray<cfloat>(const af_array &arr);
    template const Array<double>&                           getArray<double>(const af_array &arr);
    template const Array<cdouble>&                          getArray<cdouble>(const af_array &arr);
    template const Array<char>&                             getArray<char>(const af_array &arr);
    template const Array<int>&                              getArray<int>(const af_array &arr);
    template const Array<unsigned>&                         getArray<unsigned>(const af_array &arr);
    template const Array<unsigned char>&                    getArray<unsigned char>(const af_array &arr);

    template<typename T>
    Array<T> &
    getWritableArray(const af_array &arr)
    {
        const Array<T> &out = getArray<T>(arr);
        return const_cast<Array<T>&>(out);
    }

    template Array<float>&                          getWritableArray<float>(const af_array &arr);
    template Array<cfloat>&                         getWritableArray<cfloat>(const af_array &arr);
    template Array<double>&                         getWritableArray<double>(const af_array &arr);
    template Array<cdouble>&                        getWritableArray<cdouble>(const af_array &arr);
    template Array<char>&                           getWritableArray<char>(const af_array &arr);
    template Array<int>&                            getWritableArray<int>(const af_array &arr);
    template Array<unsigned>&                       getWritableArray<unsigned>(const af_array &arr);
    template Array<unsigned char>&                  getWritableArray<unsigned char>(const af_array &arr);

    template<typename T>
    af_array
    getHandle(const Array<T> &arr)
    {
        af_array out = reinterpret_cast<af_array>(&arr);
        return out;
    }

    template af_array getHandle<float>                       (const Array<float> &arr);
    template af_array getHandle<cfloat>                      (const Array<cfloat> &arr);
    template af_array getHandle<double>                      (const Array<double> &arr);
    template af_array getHandle<cdouble>                     (const Array<cdouble> &arr);
    template af_array getHandle<char>                        (const Array<char> &arr);
    template af_array getHandle<int>                         (const Array<int> &arr);
    template af_array getHandle<unsigned>                    (const Array<unsigned> &arr);
    template af_array getHandle<unsigned char>               (const Array<unsigned char> &arr);

    using af::dim4;
    template<typename T>
    Array<T>*
    createArray(const dim4 &size, const T& value)
    {
        Array<T> *out = new Array<T>(size, value);
        return out;
    }

    template<typename T>
    Array<T> *
    createView(const Array<T>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride)
    {
        assert("NOT IMPLEMENTED" && 1 != 1);
        return NULL;//createArray<T>(dims, T());
        //Array<T> *out = new Array<T>(parent, dims, offset, stride);
        //return out;
    }

    template Array<float>*          createView<float>(const Array<float>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);
    template Array<cfloat>*         createView<cfloat>(const Array<cfloat>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);
    template Array<double>*         createView<double>(const Array<double>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);
    template Array<cdouble>*        createView<cdouble>(const Array<cdouble>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);
    template Array<char>*           createView<char>(const Array<char>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);
    template Array<int>*            createView<int>(const Array<int>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);
    template Array<unsigned>*       createView<unsigned>(const Array<unsigned>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);
    template Array<unsigned char>*  createView<unsigned char>(const Array<unsigned char>& parent, const dim4 &dims, const dim4 &offset, const dim4 &stride);

    template Array<float>*                          createArray<float>(const dim4 & size, const float &value);
    template Array<cfloat>*                         createArray<cfloat>(const dim4 & size, const cfloat &value);
    template Array<double>*                         createArray<double>(const dim4 &size, const double &value);
    template Array<cdouble>*                        createArray<cdouble>(const dim4 &size, const cdouble &value);
    template Array<char>*                           createArray<char>(const dim4 &size, const char &value);
    template Array<int>*                            createArray<int>(const dim4 &size, const int &value);
    template Array<unsigned>*                       createArray<unsigned>(const dim4 &size, const unsigned &value);
    template Array<unsigned char>*                  createArray<unsigned char>(const dim4 &size, const unsigned char &value);

    template<typename T>
    void
    deleteArray(const af_array& arr)
    {
        Array<T> &obj = getWritableArray<T>(arr);
        delete &obj;
    }

    template void deleteArray<float>                        (const af_array& arr);
    template void deleteArray<cfloat>                       (const af_array& arr);
    template void deleteArray<double>                       (const af_array& arr);
    template void deleteArray<cdouble>                      (const af_array& arr);
    template void deleteArray<char>                         (const af_array& arr);
    template void deleteArray<int>                          (const af_array& arr);
    template void deleteArray<unsigned>                     (const af_array& arr);
    template void deleteArray<unsigned char>                (const af_array& arr);

}
