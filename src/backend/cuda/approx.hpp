#include <af/array.h>
#include <Array.hpp>

namespace cuda
{
    template<typename Ty, typename Tp>
    Array<Ty> *approx1(const Array<Ty> &in, const Array<Tp> &pos,
                       const af_interp_type method, const float offGrid);

    template<typename Ty, typename Tp>
    Array<Ty> *approx2(const Array<Ty> &in, const Array<Tp> &pos0, const Array<Tp> &pos1,
                       const af_interp_type method, const float offGrid);
}
