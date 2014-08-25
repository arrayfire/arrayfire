// helper functions for handling complex data
#pragma once

#include <af/defines.h>

#define upcast cuComplexFloatToDouble
#define downcast cuComplexDoubleToFloat

#define cabsf cuCabsf
#define crealf cuCrealf
#define cimagf cuCimagf
#define cdivf cuCdivf
#define cabs  cuCabs
#define creal cuCreal
#define cimag cuCimag
#define cdiv cuCdiv

#define _DH  __inline__ __device__ __host__

// FIXME: Check if we can maintain cuda namespace and use the operators

_DH float  real(cuFloatComplex c)  { return cuCrealf(c); }
_DH double real(cuDoubleComplex c) { return cuCreal(c);  }
_DH float  imag(cuFloatComplex c)  { return cuCimagf(c); }
_DH double imag(cuDoubleComplex c) { return cuCimag(c);  }

_DH float cuCreal(cuFloatComplex c) { return cuCrealf(c); }
_DH float cuCimag(cuFloatComplex c) { return cuCimagf(c); }
_DH cuFloatComplex CONJ(cuFloatComplex c) { return cuConjf(c); }
_DH cuDoubleComplex CONJ(cuDoubleComplex c) { return cuConj(c); }
_DH double CONJ(double c) { return c; }

_DH cuFloatComplex make_cuComplex(bool x)        { return make_cuComplex(x,0); }
_DH cuFloatComplex make_cuComplex(int x)         { return make_cuComplex(x,0); }
_DH cuFloatComplex make_cuComplex(unsigned x)    { return make_cuComplex(x,0); }
_DH cuFloatComplex make_cuComplex(float x)       { return make_cuComplex(x,0); }
_DH cuFloatComplex make_cuComplex(double x)      { return make_cuComplex(x,0); }
_DH cuFloatComplex make_cuComplex(cuFloatComplex x)   { return x; }
_DH cuFloatComplex make_cuComplex(cuDoubleComplex c)  { return make_cuComplex(c.x,c.y); }

_DH cuDoubleComplex make_cuDoubleComplex(bool x)        { return make_cuDoubleComplex(x,0); }
_DH cuDoubleComplex make_cuDoubleComplex(int x)         { return make_cuDoubleComplex(x,0); }
_DH cuDoubleComplex make_cuDoubleComplex(unsigned x)    { return make_cuDoubleComplex(x,0); }
_DH cuDoubleComplex make_cuDoubleComplex(float x)       { return make_cuDoubleComplex(x,0); }
_DH cuDoubleComplex make_cuDoubleComplex(double x)      { return make_cuDoubleComplex(x,0); }
_DH cuDoubleComplex make_cuDoubleComplex(cuDoubleComplex x)  { return x; }
_DH cuDoubleComplex make_cuDoubleComplex(cuFloatComplex c)   { return make_cuDoubleComplex(c.x,c.y); }

///// complex * real
_DH cuFloatComplex operator*(cuFloatComplex a, float b)    { return make_cuComplex( real(a) * b, imag(a) * b); }
_DH cuDoubleComplex operator*(cuDoubleComplex a, float b)  { return make_cuDoubleComplex(real(a) * b, imag(a) * b); }
_DH cuDoubleComplex operator*(cuFloatComplex a, double b)  { return make_cuDoubleComplex(real(a) * b, imag(a) * b); }
_DH cuDoubleComplex operator*(cuDoubleComplex a, double b) { return make_cuDoubleComplex(real(a) * b, imag(a) * b); }
_DH cuFloatComplex operator*(cuFloatComplex a, bool b)    { return a * (float)b; }
_DH cuDoubleComplex operator*(cuDoubleComplex a, bool b)    { return a * (double)b; }


_DH cuFloatComplex  operator*(float a, cuFloatComplex b)   { return b * a; }
_DH cuDoubleComplex operator*(double a, cuFloatComplex b)  { return b * a; }
_DH cuDoubleComplex operator*(float a, cuDoubleComplex b)  { return b * a; }
_DH cuDoubleComplex operator*(double a, cuDoubleComplex b) { return b * a; }


///// complex * complex
_DH cuFloatComplex   operator*(cuFloatComplex a, cuFloatComplex b)   { return cuCmulf(a,b); }
_DH cuDoubleComplex  operator*(cuDoubleComplex a, cuFloatComplex b)  { return cuCmul(a,upcast(b)); }
_DH cuDoubleComplex  operator*(cuFloatComplex a, cuDoubleComplex b)  { return cuCmul(upcast(a),b); }
_DH cuDoubleComplex  operator*(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a,b); }


///// complex - complex
_DH cuFloatComplex   operator-(cuFloatComplex a, cuFloatComplex b)   { return cuCsubf(a,b); }
_DH cuDoubleComplex  operator-(cuDoubleComplex a, cuFloatComplex b)  { return cuCsub(a,upcast(b)); }
_DH cuDoubleComplex  operator-(cuFloatComplex a, cuDoubleComplex b)  { return cuCsub(upcast(a),b); }
_DH cuDoubleComplex  operator-(cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a,b); }

//        DIV
// complex / complex
_DH static cuFloatComplex  operator/(cuFloatComplex  a, float b)   { return make_cuComplex( real(a) / b, imag(a) / (b)); }
_DH static cuDoubleComplex operator/(cuDoubleComplex a, float b)   { return make_cuDoubleComplex(real(a) / b, imag(a) / (b)); }

_DH cuFloatComplex min(cuFloatComplex a, cuFloatComplex b) { return cuCabsf(a) < cuCabsf(b) ? a : b; }
_DH cuDoubleComplex min(cuDoubleComplex a, cuDoubleComplex b) { return cuCabs(a) < cuCabs(b) ? a : b; }
_DH cuFloatComplex min(float a, cuFloatComplex b) { return fabsf(a) < cuCabsf(b) ? make_cuComplex(a) : b; }
_DH cuDoubleComplex min(double a, cuDoubleComplex b) { return fabs(a) < cuCabs(b) ? make_cuDoubleComplex(a) : b; }

_DH cuFloatComplex max(cuFloatComplex a, cuFloatComplex b) { return cuCabsf(a) > cuCabsf(b) ? a : b; }
_DH cuDoubleComplex max(cuDoubleComplex a, cuDoubleComplex b) { return cuCabs(a) > cuCabs(b) ? a : b; }
_DH cuFloatComplex max(float a, cuFloatComplex b) { return fabsf(a) > cuCabsf(b) ? make_cuComplex(a) : b; }
_DH cuDoubleComplex max(double a, cuDoubleComplex b) { return fabs(a) > cuCabs(b) ? make_cuDoubleComplex(a) : b; }

#define NEQ_OPS(OP, OT)                                     \
    _DH static bool operator OP(cuFloatComplex a, float b)       \
    { return (imag(a) OP 0 OT real(a) OP b); }              \
    _DH static bool operator OP(cuDoubleComplex a, double b)     \
    { return (imag(a) OP 0 OT real(a) OP b); }              \
    _DH static bool operator OP(float b, cuFloatComplex a)       \
    { return (imag(a) OP 0 OT real(a) OP b); }              \
    _DH static bool operator OP(double b, cuDoubleComplex a)     \
    { return (imag(a) OP 0 OT real(a) OP b); }              \
    _DH static bool operator OP(cuFloatComplex a, cuFloatComplex b)   \
    { return (imag(a) OP imag(b) OT real(a) OP real(b)); }  \
    _DH static bool operator OP(cuDoubleComplex a, cuDoubleComplex b) \
    { return (imag(a) OP imag(b) OT real(a) OP real(b)); }

NEQ_OPS(==, &&);
NEQ_OPS(!=, ||);

_DH bool operator*(bool a, cuFloatComplex b)    { return a & (b != 0); }
_DH bool operator*(bool a, cuDoubleComplex b)    { return a & (b != 0); }
_DH bool operator+(bool a, cuFloatComplex b)    { return a | (b != 0); }
_DH bool operator+(bool a, cuDoubleComplex b)    { return a | (b != 0); }

// from muxa_cplx.h
// complex *= complex
_DH static cuFloatComplex operator *=(cuFloatComplex &x, cuFloatComplex y){
    float re = cuCrealf(x)*cuCrealf(y) - cuCimagf(x)*cuCimagf(y);
    float im = cuCrealf(x)*cuCimagf(y) + cuCimagf(x)*cuCrealf(y);
    x.x = re;
    x.y = im;
    return x;
}

//        DIV
// complex / complex
_DH static cuFloatComplex operator/(cuFloatComplex a, cuFloatComplex b)   { return cuCdivf(a, b); }

// complex / cuCrealf (inv)
_DH static cuFloatComplex operator/(float a, cuFloatComplex b)        { return make_cuComplex(a, 0.0f) / b; }

// complex /= complex
_DH static cuFloatComplex operator /=(cuFloatComplex &x, cuFloatComplex y){
    float s = fabs(cuCrealf(y)) + fabs(cuCimagf(y));
    float oos = 1.0f / s;
    float xrs = cuCrealf(x)*oos;
    float xis = cuCimagf(x)*oos;
    float yrs = cuCrealf(y)*oos;
    float yis = cuCimagf(y)*oos;
    s = yrs*yrs + yis*yis;
    oos = 1.0f / s;
    x.x = (xrs*yrs + xis*yis)*oos;
    x.y = (xis*yrs - xrs*yis)*oos;
    return x;
}

//        ADD
// complex + cuCrealf
_DH static cuFloatComplex operator+(cuFloatComplex a, float b)        { return make_cuComplex(cuCrealf(a) + b, cuCimagf(a)); }
_DH static cuFloatComplex operator+(float a, cuFloatComplex b)        { return b + a; }

// HACK
_DH static cuFloatComplex operator+(cuFloatComplex a, bool b)    { return a; }
_DH static cuDoubleComplex operator+(cuDoubleComplex a, bool b)    { return a; }

// complex + complex
_DH static cuFloatComplex operator+(cuFloatComplex a, cuFloatComplex b)   { return cuCaddf(a, b); }

// complex += complex
_DH static cuFloatComplex operator +=(cuFloatComplex &x, cuFloatComplex y){
    x.x = cuCrealf(x) + cuCrealf(y);
    x.y = cuCimagf(x) + cuCimagf(y);
    return x;
}

// +complex
_DH static cuFloatComplex operator +(cuFloatComplex x)                { return x; }

//        SUB
// -complex
_DH static cuFloatComplex operator -(cuFloatComplex x)                 { return make_cuComplex(-cuCrealf(x), -cuCimagf(x)); }

// complex - cuCrealf (neg)
_DH static cuFloatComplex operator-(cuFloatComplex a, float b)         { return make_cuComplex(cuCrealf(a) - b, cuCimagf(a)); }
_DH static cuFloatComplex operator-(float a, cuFloatComplex b)         { return -b + a; }

// complex - complex
//_DH static cuFloatComplex operator-(cuFloatComplex a, cuFloatComplex b)    { return cuCsubf(a, b); }

// complex -= complex
_DH static cuFloatComplex operator -=(cuFloatComplex &x, cuFloatComplex y){
    x.x = cuCrealf(x) - cuCrealf(y);
    x.y = cuCimagf(x) - cuCimagf(y);
    return x;
}


//        MUL
// complex * complex
_DH static cuDoubleComplex operator *=(cuDoubleComplex &x, cuDoubleComplex y){
    double re = cuCreal(x)*cuCreal(y) - cuCimag(x)*cuCimag(y);
    double im = cuCreal(x)*cuCimag(y) + cuCimag(x)*cuCreal(y);
    x.x = re;
    x.y = im;
    return x;
}
_DH static cuDoubleComplex operator *=(cuDoubleComplex &x, cuFloatComplex y) { return x *= upcast(y); }

//        DIV
// complex / complex
_DH static cuDoubleComplex operator/(cuDoubleComplex a, cuDoubleComplex b)    { return cuCdiv(a, b); }
_DH static cuDoubleComplex operator/(cuDoubleComplex a, cuFloatComplex b)    { return a/upcast(b); }
_DH static cuDoubleComplex operator/(cuFloatComplex a, cuDoubleComplex b)    { return upcast(a)/b; }

// complex / cuCreal (inv)
_DH static cuDoubleComplex operator/(cuFloatComplex a, double b)        { return upcast(a) / make_cuDoubleComplex(b, 0.0); }
_DH static cuDoubleComplex operator/(cuDoubleComplex a, double b)        { return a / make_cuDoubleComplex(b, 0.0); }
_DH static cuDoubleComplex operator/(float a, cuDoubleComplex b)         { return upcast(make_cuComplex(a, 0.0f)) / b; }
_DH static cuDoubleComplex operator/(double a, cuFloatComplex b)        { return make_cuDoubleComplex(a, 0.0) / upcast(b); }
_DH static cuDoubleComplex operator/(double a, cuDoubleComplex b)        { return make_cuDoubleComplex(a, 0.0) / b; }

// complex /= complex
_DH static cuDoubleComplex operator /=(cuDoubleComplex &x, cuDoubleComplex y) {
    double s = fabs(cuCreal(y)) +
              fabs(cuCimag(y));
    double oos = 1.0 / s;
    double xrs = cuCreal(x)*oos;
    double xis = cuCimag(x)*oos;
    double yrs = cuCreal(y)*oos;
    double yis = cuCimag(y)*oos;
    s = yrs*yrs + yis*yis;
    oos = 1.0 / s;
    x.x = (xrs*yrs + xis*yis)*oos;
    x.y = (xis*yrs - xrs*yis)*oos;
    return x;
}
_DH static cuDoubleComplex operator /=(cuDoubleComplex &x, cuFloatComplex y)  { return x /= upcast(y); }

//        ADD
// complex + cuCreal
_DH static cuDoubleComplex operator+(cuDoubleComplex a, float b)          { return make_cuDoubleComplex(cuCreal(a) + b, cuCimag(a)); }
_DH static cuDoubleComplex operator+(cuFloatComplex a, double b)         { return make_cuDoubleComplex(cuCrealf(a) + b, cuCimagf(a)); }
_DH static cuDoubleComplex operator+(cuDoubleComplex a, double b)         { return make_cuDoubleComplex(cuCreal(a) + b, cuCimag(a)); }
_DH static cuDoubleComplex operator+(float a, cuDoubleComplex b)          { return b + a; }
_DH static cuDoubleComplex operator+(double a, cuFloatComplex b)         { return b + a; }
_DH static cuDoubleComplex operator+(double a, cuDoubleComplex b)         { return b + a; }

// complex + complex
_DH static cuDoubleComplex operator+(cuDoubleComplex a, cuFloatComplex b)     { return cuCadd(a, upcast(b)); }
_DH static cuDoubleComplex operator+(cuFloatComplex a, cuDoubleComplex b)     { return cuCadd(upcast(a), b); }
_DH static cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b)     { return cuCadd(a, b); }

// complex += complex
_DH static cuDoubleComplex operator +=(cuDoubleComplex &x, cuDoubleComplex y)  {
    x.x = cuCreal(x) + cuCreal(y);
    x.y = cuCimag(x) + cuCimag(y);
    return x;
}
_DH static cuDoubleComplex operator +=(cuDoubleComplex &x, cuFloatComplex y)  { return x += upcast(y); }
_DH static cuFloatComplex operator +=(cuFloatComplex &x, cuDoubleComplex y)  { return x += downcast(y); }

// +complex
_DH static cuDoubleComplex operator +(cuDoubleComplex &x)                 { return x; }

//        SUB
// -complex
_DH static cuDoubleComplex operator -(cuDoubleComplex x)                  { return make_cuDoubleComplex(-cuCreal(x), -cuCimag(x)); }

// complex - cuCreal (neg)
_DH static cuDoubleComplex operator-(cuDoubleComplex a, float b)          { return make_cuDoubleComplex(cuCreal(a) - b, cuCimag(a)); }
_DH static cuDoubleComplex operator-(cuFloatComplex a, double b)         { return make_cuDoubleComplex(cuCrealf(a) - b, cuCimagf(a)); }
_DH static cuDoubleComplex operator-(cuDoubleComplex a, double b)         { return make_cuDoubleComplex(cuCreal(a) - b, cuCimag(a)); }
_DH static cuDoubleComplex operator-(float a, cuDoubleComplex b)          { return -b + a; }
_DH static cuDoubleComplex operator-(double a, cuFloatComplex b)         { return -b + a; }
_DH static cuDoubleComplex operator-(double a, cuDoubleComplex b)         { return -b + a; }

// complex -= complex
_DH static cuDoubleComplex operator -=(cuDoubleComplex &x, cuDoubleComplex y)  {
    x.x = cuCreal(x) - cuCreal(y);
    x.y = cuCimag(x) - cuCimag(y);
    return x;
}
_DH static cuDoubleComplex operator -=(cuDoubleComplex &x, cuFloatComplex y)  { return x -= upcast(y); }

#define D_i            make_cuDoubleComplex(0.0, 1.0)
#define F_i            make_cuComplex(1.0, 0.0)

static inline double cuCabs(cuFloatComplex x) { return cuCabsf(x); }
static inline double cuCreal(double x)   { return x; }
static inline double cuCimag(double x)   { return 0; }

#undef _DH
