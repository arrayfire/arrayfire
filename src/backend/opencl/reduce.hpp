#include <af/array.h>
#include <Array.hpp>
#include "../ops.hpp"

namespace opencl
{
    template<af_op_t op, typename Ti, typename To>
    Array<To>* reduce(const Array<Ti> in, const int dim);
}
