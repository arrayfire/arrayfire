
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
    print(const Array<T> &A)
    {
        std::cout << A;
    }

#define INSTANTIATE(T)                          \
    template void print<T> (const Array<T> &A); \

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
    INSTANTIATE(char)
}
