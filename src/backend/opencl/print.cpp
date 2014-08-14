
#include <iostream>
#include <cassert>
#include <print.hpp>
#include <Array.hpp>

namespace opencl
{
    template<typename T>
    void
    printer(ostream &out, const cl::Buffer &ptr, const Array<T> &info, unsigned dim)
    {
        assert("NOT IMPLEMENTED" && 1 != 1);
    }

    template<typename T>
    ostream&
    operator <<(ostream &out, const Array<T> &arr)
    {
        out << "TRANSPOSED\n";
        out << "Dim:" << arr.dims();
        out << "Offset: " << arr.offsets();
        out << "Stride: " << arr.strides();
        printer(out, arr.get(), arr, arr.ndims() - 1);
        return out;
    }

    template<typename T>
    void
    print(const af_array &arr) {
        const Array<T> &impl = getArray<T>(arr);
        std::cout << impl;
    }

    template void print<float>                   (const af_array &arr);
    template void print<cfloat>                  (const af_array &arr);
    template void print<double>                  (const af_array &arr);
    template void print<cdouble>                 (const af_array &arr);
    template void print<char>                    (const af_array &arr);
    template void print<int>                     (const af_array &arr);
    template void print<unsigned>                (const af_array &arr);
    template void print<uchar>                   (const af_array &arr);
}
