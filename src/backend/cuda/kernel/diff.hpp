#include <af/defines.h>

namespace cuda
{
namespace kernel
{
    template<typename T, unsigned dim, bool isDiff2>
    void diff(T *out, const T *in,
               const unsigned oElem, const unsigned ondims, const dim_type *odims, const dim_type *ostrides,
               const unsigned iElem, const unsigned indims, const dim_type *idims, const dim_type *istrides);

}
}
