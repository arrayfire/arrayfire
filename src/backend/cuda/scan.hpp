#include <af/array.h>
#include <Array.hpp>
#include "../ops.hpp"

namespace cuda
{
    template<af_op_t op, typename Ti, typename To>
    Array<To>* scan(const Array<Ti>& in, const int dim);
}
