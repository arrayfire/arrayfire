#include <cassert>
#include <af/array.h>
#include <Array.hpp>
#include <moddims.hpp>

namespace opencl
{
    template<typename T>
    af_array moddims(const af_array &arr, const af::dim4 &newDims)
    {
        const Array<T> &inArray = getArray<T>(arr);
        assert("Indexed Arrays not supported yet in opencl" && inArray.isOwner()==true);
        Array<T> *out = createValueArray<T>(newDims,(T)0);

        //FIXME: Add checks and possibly move the copy to handler
        getQueue(0).enqueueCopyBuffer(inArray.get(),out->get(),0,0,inArray.elements()*sizeof(T));

        return getHandle(*out);
    }

    template af_array moddims<float>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<double>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<char>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<int>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<unsigned>(const af_array &arr, const af::dim4 &newDims);
    template af_array moddims<unsigned char>(const af_array &arr, const af::dim4 &newDims);

    template<>
    af_array moddims<cfloat>(const af_array &arr, const af::dim4 &newDims)
    {
        assert("moddims not supported for cfloat type" && 1==1);
    }

    template<>
    af_array moddims<cdouble>(const af_array &arr, const af::dim4 &newDims)
    {
        assert("moddims not supported for cdouble type" && 1==1);
    }
}
