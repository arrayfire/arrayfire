/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <af/defines.h>

typedef struct af_seq {
    double begin, end;
    double step;
} af_seq;

static const af_seq af_span = {1, 1, 0};

#ifdef __cplusplus
namespace af
{
class AFAPI seq
{
public:
    af_seq s;
    size_t size;

    seq(double = 0);
    ~seq();

    // begin, end, step
    seq(double begin, double end, double step = 1);

    seq(const af_seq& s_);

    seq& operator=(const af_seq& s);

    inline seq operator-()         { return seq(-s.begin, -s.end,  -s.step); }

    inline seq operator+(double x) { return seq(s.begin + x, s.end + x, s.step); }

    inline seq operator-(double x) { return seq(s.begin - x, s.end - x, s.step); }

    inline seq operator*(double x) { return seq(s.begin * x, s.end * x, s.step * x); }

    friend inline seq operator+(double x, seq y) { return  y + x; }

    friend inline seq operator-(double x, seq y) { return -y + x; }

    friend inline seq operator*(double x, seq y) { return  y * x; }

    private:
    void init(double begin, double end, double step);
};

extern AFAPI int end;
extern AFAPI seq span;

}
#endif
