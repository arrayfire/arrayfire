#include <af/array.h>
#include <Array.hpp>
#include <iosfwd>

namespace opencl
{
    using std::ostream;

    template<typename T>
    ostream&
    operator <<(ostream &out, const Array<T> &arr);

    template<typename T>
    void
    print(const Array<T> &A);
}
