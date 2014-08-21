#include <af/defines.h>

namespace cuda
{
namespace kernel
{
    template<typename T>
    void diff1(T *out, const T *in, const unsigned dim,
               const unsigned oElem, const unsigned ondims, const dim_type *odims, const dim_type *ostrides,
               const unsigned iElem, const unsigned indims, const dim_type *idims, const dim_type *istrides);

    template<typename T>
    void diff2(T *out, const T *in, const unsigned dim,
               const unsigned oElem, const unsigned ondims, const dim_type *odims, const dim_type *ostrides,
               const unsigned iElem, const unsigned indims, const dim_type *idims, const dim_type *istrides);

}
}
