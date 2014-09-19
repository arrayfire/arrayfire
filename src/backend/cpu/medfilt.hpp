#include <Array.hpp>

namespace cpu
{

template<typename T, af_pad_type edge_pad>
Array<T> * medfilt(const Array<T> &in, dim_type w_len, dim_type w_wid);

}
