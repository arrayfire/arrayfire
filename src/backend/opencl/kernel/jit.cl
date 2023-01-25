/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define __select(cond, a, b) (cond) ? (a) : (b)
#define __not_select(cond, a, b) (cond) ? (b) : (a)
#define __circular_mod(a, b) ((a) < (b)) ? (a) : (a - b)

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

#define __crealf(in) ((in).x)
#define __cimagf(in) ((in).y)
#define __cabsf(in) hypot((in).x, (in).y)

#define __creal(in) ((in).x)
#define __cimag(in) ((in).y)
#define __cabs(in) hypot((in).x, (in).y)
#define __sigmoid(in) (1.0 / (1 + exp(-(in))))

float2 __cconjf(float2 in) {
    float2 out = {in.x, -in.y};
    return out;
}

float2 __caddf(float2 lhs, float2 rhs) {
    float2 out = {lhs.x + rhs.x, lhs.y + rhs.y};
    return out;
}

float2 __csubf(float2 lhs, float2 rhs) {
    float2 out = {lhs.x - rhs.x, lhs.y - rhs.y};
    return out;
}

float2 __cmulf(float2 lhs, float2 rhs) {
    float2 out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}

// FIXME: overflow / underflow issues
float2 __cdivf(float2 lhs, float2 rhs) {
    // Normalize by absolute value and multiply
    float rhs_abs     = __cabsf(rhs);
    float inv_rhs_abs = 1.0f / rhs_abs;
    float rhs_x       = inv_rhs_abs * rhs.x;
    float rhs_y       = inv_rhs_abs * rhs.y;
    float2 out = {lhs.x * rhs_x + lhs.y * rhs_y, lhs.y * rhs_x - lhs.x * rhs_y};
    out.x *= inv_rhs_abs;
    out.y *= inv_rhs_abs;
    return out;
}

#define __candf(lhs, rhs) __cabsf(lhs) && __cabsf(rhs)
#define __cand(lhs, rhs) __cabs(lhs) && __cabs(rhs)

#define __corf(lhs, rhs) __cabsf(lhs) || __cabsf(rhs)
#define __cor(lhs, rhs) __cabs(lhs) || __cabs(rhs)

#define __ceqf(lhs, rhs) (((lhs).x == (rhs).x) && ((lhs).y == (rhs).y))
#define __cneqf(lhs, rhs) !__ceqf((lhs), (rhs))
#define __cltf(lhs, rhs) (__cabsf(lhs) < __cabsf(rhs))
#define __clef(lhs, rhs) (__cabsf(lhs) <= __cabsf(rhs))
#define __cgtf(lhs, rhs) (__cabsf(lhs) > __cabsf(rhs))
#define __cgef(lhs, rhs) (__cabsf(lhs) >= __cabsf(rhs))

#define __ceq(lhs, rhs) (((lhs).x == (rhs).x) && ((lhs).y == (rhs).y))
#define __cneq(lhs, rhs) !__ceq((lhs), (rhs))
#define __clt(lhs, rhs) (__cabs(lhs) < __cabs(rhs))
#define __cle(lhs, rhs) (__cabs(lhs) <= __cabs(rhs))
#define __cgt(lhs, rhs) (__cabs(lhs) > __cabs(rhs))
#define __cge(lhs, rhs) (__cabs(lhs) >= __cabs(rhs))

#define __bitnot(in) (~(in))
#define __bitor(lhs, rhs) ((lhs) | (rhs))
#define __bitand(lhs, rhs) ((lhs) & (rhs))
#define __bitxor(lhs, rhs) ((lhs) ^ (rhs))
#define __bitshiftl(lhs, rhs) ((lhs) << (rhs))
#define __bitshiftr(lhs, rhs) ((lhs) >> (rhs))

#define __min(lhs, rhs) ((lhs) < (rhs)) ? (lhs) : (rhs)
#define __max(lhs, rhs) ((lhs) > (rhs)) ? (lhs) : (rhs)
#define __rem(lhs, rhs) ((lhs) % (rhs))
#define __mod(lhs, rhs) ((lhs) % (rhs))

#define __pow(lhs, rhs)                                                 \
    convert_int_rte(pow(convert_float_rte(lhs), convert_float_rte(rhs)))
#ifdef USE_DOUBLE
#define __powll(lhs, rhs) \
    convert_long_rte(pow(convert_double_rte(lhs), convert_double_rte(rhs)))
#define __powul(lhs, rhs) \
    convert_ulong_rte(pow(convert_double_rte(lhs), convert_double_rte(rhs)))
#else
#define __powll(lhs, rhs) \
    convert_long_rte(pow(convert_float_rte(lhs), convert_float_rte(rhs)))
#define __powul(lhs, rhs) \
    convert_ulong_rte(pow(convert_float_rte(lhs), convert_float_rte(rhs)))
#endif

#ifdef USE_DOUBLE
#define __powui(lhs, rhs) \
    convert_uint_rte(pow(convert_double_rte(lhs), convert_double_rte(rhs)))
#define __powsi(lhs, rhs) \
    convert_int_rte(pow(convert_double_rte(lhs), convert_double_rte(rhs)))
#else
#define __powui(lhs, rhs) \
    convert_uint_rte(pow(convert_float_rte(lhs), convert_float_rte(rhs)))
#define __powsi(lhs, rhs) \
    convert_int_rte(pow(convert_float_rte(lhs), convert_float_rte(rhs)))
#endif

float2 __cminf(float2 lhs, float2 rhs) {
    return __cabsf(lhs) < __cabsf(rhs) ? lhs : rhs;
}

float2 __cmaxf(float2 lhs, float2 rhs) {
    return __cabsf(lhs) > __cabsf(rhs) ? lhs : rhs;
}

float2 __cplx2f(float lhs, float rhs) {
    float2 out = {lhs, rhs};
    return out;
}

float2 __convert_cfloat(float in) {
    float2 out = {in, 0};
    return out;
}

#define __convert_char(val) (char)(convert_char((val)) != 0)

#define frem(lhs, rhs) remainder((lhs), (rhs))

#define iszero(a) ((a) == 0)

float2 __convert_c2c(float2 in) { return in; }

#ifdef USE_DOUBLE

float2 __convert_z2c(double2 in) {
    float2 out = {in.x, in.y};
    return out;
}

double2 __cconj(double2 in) {
    double2 out = {in.x, -in.y};
    return out;
}

double2 __cadd(double2 lhs, double2 rhs) {
    double2 out = {lhs.x + rhs.x, lhs.y + rhs.y};
    return out;
}

double2 __csub(double2 lhs, double2 rhs) {
    double2 out = {lhs.x - rhs.x, lhs.y - rhs.y};
    return out;
}

double2 __cmul(double2 lhs, double2 rhs) {
    double2 out;
    out.x = lhs.x * rhs.x - lhs.y * rhs.y;
    out.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return out;
}

double2 __cdiv(double2 lhs, double2 rhs) {
    // Normalize by absolute value and multiply
    double rhs_abs     = __cabs(rhs);
    double inv_rhs_abs = 1.0 / rhs_abs;
    double rhs_x       = inv_rhs_abs * rhs.x;
    double rhs_y       = inv_rhs_abs * rhs.y;
    double2 out        = {lhs.x * rhs_x + lhs.y * rhs_y,
                   lhs.y * rhs_x - lhs.x * rhs_y};
    out.x *= inv_rhs_abs;
    out.y *= inv_rhs_abs;
    return out;
}

double2 __cmin(double2 lhs, double2 rhs) {
    return __cabs(lhs) < __cabs(rhs) ? lhs : rhs;
}

double2 __cmax(double2 lhs, double2 rhs) {
    return __cabs(lhs) > __cabs(rhs) ? lhs : rhs;
}

double2 __cplx2(double lhs, double rhs) {
    double2 out = {lhs, rhs};
    return out;
}

double2 __convert_cdouble(double in) {
    double2 out = {in, 0};
    return out;
}

double2 __convert_c2z(float2 in) {
    double2 out = {in.x, in.y};
    return out;
}

double2 __convert_z2z(double2 in) { return in; }

#endif  // USE_DOUBLE
