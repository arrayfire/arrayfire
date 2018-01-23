/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

typedef float2 cuFloatComplex;
typedef cuFloatComplex cfloat;

typedef double2 cuDoubleComplex;
typedef cuDoubleComplex cdouble;

// ----------------------------------------------
// COMMON OPERATIONS
// ----------------------------------------------

#define __select(cond, a, b) (cond) ? (a) : (b)
#define __not_select(cond, a, b) (cond) ? (b) : (a)

// ----------------------------------------------
// REAL NUMBER OPERATIONS
// ----------------------------------------------
#define sign(in) signbit((in))
#define __noop(a) (a)
#define __add(lhs, rhs) (lhs) + (rhs)
#define __sub(lhs, rhs) (lhs) - (rhs)
#define __mul(lhs, rhs) (lhs) * (rhs)
#define __div(lhs, rhs) (lhs) / (rhs)
#define __and(lhs, rhs) (lhs) && (rhs)
#define __or(lhs, rhs) (lhs) || (rhs)

#define __lt(lhs, rhs) (lhs) < (rhs)
#define __gt(lhs, rhs) (lhs) > (rhs)
#define __le(lhs, rhs) (lhs) <= (rhs)
#define __ge(lhs, rhs) (lhs) >= (rhs)
#define __eq(lhs, rhs) (lhs) == (rhs)
#define __neq(lhs, rhs) (lhs) != (rhs)

#define __conj(in) (in)
#define __real(in) (in)
#define __imag(in) (0)
#define __abs(in) abs(in)
#define __sigmoid(in) (1.0/(1 + exp(-(in))))

#define __bitor(lhs, rhs) ((lhs) | (rhs))
#define __bitand(lhs, rhs) ((lhs) & (rhs))
#define __bitxor(lhs, rhs) ((lhs) ^ (rhs))
#define __bitshiftl(lhs, rhs) ((lhs) << (rhs))
#define __bitshiftr(lhs, rhs) ((lhs) >> (rhs))

#define __min(lhs, rhs) ((lhs) < (rhs)) ? (lhs) : (rhs)
#define __max(lhs, rhs) ((lhs) > (rhs)) ? (lhs) : (rhs)
#define __rem(lhs, rhs) ((lhs) % (rhs))
#define __mod(lhs, rhs) ((lhs) % (rhs))
#define __pow(lhs, rhs) fpow((float)lhs, (float)rhs)

#define __convert_char(val) (char)((val) != 0)
#define fpow(lhs, rhs) pow((lhs), (rhs))
#define frem(lhs, rhs) remainder((lhs), (rhs))
#define iszero(a) ((a) == 0)

// ----------------------------------------------
// COMPLEX FLOAT OPERATIONS
// ----------------------------------------------

#define __crealf(in) ((in).x)
#define __cimagf(in) ((in).y)
#define __cabsf(in) hypotf(in.x, in.y)

__device__ cfloat __cplx2f(float x, float y)
{
    cfloat res = {x, y};
    return res;
}

__device__ cfloat __cconjf(cfloat in)
{
    cfloat res = {in.x, -in.y};
    return res;
}

__device__ cfloat __caddf(cfloat lhs, cfloat rhs)
{
    cfloat res = {lhs.x + rhs.x, lhs.y + rhs.y};
    return res;
}

__device__ cfloat __csubf(cfloat lhs, cfloat rhs)
{
    cfloat res = {lhs.x - rhs.x, lhs.y - rhs.y};
    return res;
}

__device__ cfloat __cmulf(cfloat lhs, cfloat rhs)
{
    cfloat out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}

__device__ cfloat __cdivf(cfloat lhs, cfloat rhs)
{
    // Normalize by absolute value and multiply
    float rhs_abs = __cabsf(rhs);
    float inv_rhs_abs = 1.0f / rhs_abs;
    float rhs_x = inv_rhs_abs * rhs.x;
    float rhs_y = inv_rhs_abs * rhs.y;
    cfloat out = {lhs.x * rhs_x + lhs.y * rhs_y,
                  lhs.y * rhs_x - lhs.x * rhs_y};
    out.x *= inv_rhs_abs;
    out.y *= inv_rhs_abs;
    return out;
}

__device__ cfloat __cminf(cfloat lhs, cfloat rhs)
{
    return __cabsf(lhs) < __cabsf(rhs) ? lhs : rhs;
}

__device__ cfloat __cmaxf(cfloat lhs, cfloat rhs)
{
    return __cabsf(lhs) > __cabsf(rhs) ? lhs : rhs;
}
#define __candf(lhs, rhs) __cabsf(lhs) && __cabsf(rhs)
#define __corf(lhs, rhs) __cabsf(lhs) || __cabsf(rhs)
#define __ceqf(lhs, rhs) (((lhs).x == (rhs).x) && ((lhs).y == (rhs).y))
#define __cneqf(lhs, rhs) !__ceqf((lhs), (rhs))
#define __cltf(lhs, rhs) (__cabsf(lhs) < __cabsf(rhs))
#define __clef(lhs, rhs) (__cabsf(lhs) <= __cabsf(rhs))
#define __cgtf(lhs, rhs) (__cabsf(lhs) > __cabsf(rhs))
#define __cgef(lhs, rhs) (__cabsf(lhs) >= __cabsf(rhs))
#define __convert_cfloat(real) __cplx2f(real, 0)
#define __convert_c2c(in) (in)
#define __convert_z2c(in) __cplx2f((float)in.x, (float)in.y)

// ----------------------------------------------
// COMPLEX DOUBLE OPERATIONS
// ----------------------------------------------
#define __creal(in) ((in).x)
#define __cimag(in) ((in).y)
#define __cabs(in) hypot(in.x, in.y)

__device__ cdouble __cplx2(double x, double y)
{
    cdouble res = {x, y};
    return res;
}

__device__ cdouble __cconj(cdouble in)
{
    cdouble res = {in.x, -in.y};
    return res;
}

__device__ cdouble __cadd(cdouble lhs, cdouble rhs)
{
    cdouble res = {lhs.x + rhs.x, lhs.y + rhs.y};
    return res;
}

__device__ cdouble __csub(cdouble lhs, cdouble rhs)
{
    cdouble res = {lhs.x - rhs.x, lhs.y - rhs.y};
    return res;
}

__device__ cdouble __cmul(cdouble lhs, cdouble rhs)
{
    cdouble out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}

__device__ cdouble __cdiv(cdouble lhs, cdouble rhs)
{
    // Normalize by absolute value and multiply
    double rhs_abs = __cabs(rhs);
    double inv_rhs_abs = 1.0 / rhs_abs;
    double rhs_x = inv_rhs_abs * rhs.x;
    double rhs_y = inv_rhs_abs * rhs.y;
    cdouble out = {lhs.x * rhs_x + lhs.y * rhs_y,
                   lhs.y * rhs_x - lhs.x * rhs_y};
    out.x *= inv_rhs_abs;
    out.y *= inv_rhs_abs;
    return out;
}

__device__ cdouble __cmin(cdouble lhs, cdouble rhs)
{
    return __cabs(lhs) < __cabs(rhs) ? lhs : rhs;
}

__device__ cdouble __cmax(cdouble lhs, cdouble rhs)
{
    return __cabs(lhs) > __cabs(rhs) ? lhs : rhs;
}
#define __cand(lhs, rhs) __cabs(lhs) && __cabs(rhs)
#define __cor(lhs, rhs) __cabs(lhs) || __cabs(rhs)
#define __ceq(lhs, rhs) (((lhs).x == (rhs).x) && ((lhs).y == (rhs).y))
#define __cneq(lhs, rhs) !__ceq((lhs), (rhs))
#define __clt(lhs, rhs) (__cabs(lhs) < __cabs(rhs))
#define __cle(lhs, rhs) (__cabs(lhs) <= __cabs(rhs))
#define __cgt(lhs, rhs) (__cabs(lhs) > __cabs(rhs))
#define __cge(lhs, rhs) (__cabs(lhs) >= __cabs(rhs))
#define __convert_cdouble(real) __cplx2(real, 0)
#define __convert_z2z(in) (in)
#define __convert_c2z(in) __cplx2((double)in.x, (double)in.y)
