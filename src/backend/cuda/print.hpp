#include <af/array.h>
#include <Array.hpp>
#include <iosfwd>
#include <backend.hpp>

namespace cuda
{
    std::ostream&
    operator<<(std::ostream &out, const cfloat& var);

    std::ostream&
    operator<<(std::ostream &out, const cdouble& var);

    template<typename T>
    void
    print(const Array<T> &A);
}
