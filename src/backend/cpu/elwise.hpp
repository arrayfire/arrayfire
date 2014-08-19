#include <af/defines.h>
#include <af/array.h>
#include <Array.hpp>

namespace cpu
{
    template<typename Tl, typename Tr, typename To, typename Op>
    Array<To>* binOp(const Array<Tl> &lhs, const Array<Tr> &rhs);
}
