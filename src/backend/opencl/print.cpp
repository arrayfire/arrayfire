
#include <iostream>
#include <cassert>
#include <print.hpp>
#include <Array.hpp>
#include <copy.hpp>

namespace opencl
{
    using std::ostream;

    ostream&
    operator<<(ostream &out, const cfloat& var)
    {
        out << var.s[0] << " " << var.s[1] << "i";
        return out;
    }

    ostream&
    operator<<(ostream &out, const cdouble& var)
    {
        out << var.s[0] << " " << var.s[1] << "i";
        return out;
    }

    ostream&
    operator<<(ostream &out, const unsigned char& var)
    {
        out << (int)var;
        return out;
    }


    template<typename T>
    void printer(ostream &out, const T* ptr, const Array<T> &info, unsigned dim)
    {
        dim_type stride =   info.strides()[dim];
        dim_type d      =   info.dims()[dim];

        if(dim == 0) {
            for(dim_type i = 0, j = 0; i < d; i++, j+=stride) {
                out << ptr[j] << "\t";
            }
            out << std::endl;
        }
        else {
            for(dim_type i = 0; i < d; i++) {
                printer(out, ptr, info, dim - 1);
                ptr += stride;
            }
        }
    }

    template<typename T>
    ostream&
    operator <<(ostream &out, const Array<T> &arr)
    {
        out << "TRANSPOSED\n";
        out << "Dim:" << arr.dims();
        out << "Offset: " << arr.offsets();
        out << "Stride: " << arr.strides();
        T * dataPtr = new T[arr.elements()];
        copyData(dataPtr,arr);
        printer(out, dataPtr, arr, arr.ndims() - 1);
        delete[] dataPtr;
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
