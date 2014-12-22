#include <af/seq.h>
#include <af/array.h>
#include <af/data.h>
#include "error.hpp"

namespace af
{

AFAPI int end = -1;
AFAPI seq span(af_span);

void seq::init(double begin, double end, double step)
{
    this->s.begin = begin;
    this->s.end   = end;
    this->s.step  = step;
    if(step != 0) {       // Not Span
        size = fabs((end - begin) / step) + 1;
    } else {
        size = 0;
    }
}

#ifndef signbit
// wtf windows?!
inline int signbit(double x)
{
    if (x < 0) return -1;
    return  0;
}
#endif

seq::~seq()
{
}

seq::seq(double n)
{
    if (n == end) {
        init(-1, -1, 0);
    } else {
        init(0, n - 1, 1);
    }
}

seq::seq(const af_seq& s_)
{
    init(s_.begin, s_.end, s_.step);
}

seq& seq::operator=(const af_seq& s_)
{
    init(s_.begin, s_.end, s_.step);
    return *this;
}

seq::seq(double begin, double end, double step)
{
    if(begin == -1 && end == -1) {
        step = 0;           // end
    }

    if (step == 0) {
        if (begin != end)   // Span
            AF_THROW(AF_ERR_INVALID_ARG);
    }
    if (end >= 0 && begin >= 0 && signbit(end-begin) != signbit(step))
        AF_THROW(AF_ERR_INVALID_ARG);
        //AF_THROW("step must match direction of sequence");
    init(begin, end, step);
}

seq::operator array() const
{
    dim_type len = (int)((s.end - s.begin + 1) / s.step);
    array res = s.begin + s.step * iota(len);
    return res;
}

}
