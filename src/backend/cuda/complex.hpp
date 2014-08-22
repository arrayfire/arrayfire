// helper functions for handling complex data
#pragma once

#include <af/defines.h>
#include <backend.hpp>

// FIXME: Remove this assert when 292 is fixed
#include <cassert>

#define make_cfloat make_cuComplex
#define make_cdouble make_cuDoubleComplex
#define upcast cuComplexFloatToDouble
#define downcast cuComplexDoubleToFloat
#define make_cuDoubleComplex   make_cuDoubleComplex

#define cabsf cuCabsf
#define crealf cuCrealf
#define cimagf cuCimagf
#define cdivf cuCdivf
#define cabs  cuCabs
#define creal cuCreal
#define cimag cuCimag
#define cdiv cuCdiv

#define _DH  __inline__ __device__ __host__

namespace cuda
{

_DH float  real(cfloat c)  { return cuCrealf(c); }
_DH double real(cdouble c) { return cuCreal(c);  }
_DH float  imag(cfloat c)  { return cuCimagf(c); }
_DH double imag(cdouble c) { return cuCimag(c);  }

_DH float cuCreal(cfloat c) { return cuCrealf(c); }
_DH float cuCimag(cfloat c) { return cuCimagf(c); }
_DH cfloat CONJ(cfloat c) { return cuConjf(c); }
_DH cdouble CONJ(cdouble c) { return cuConj(c); }
_DH double CONJ(double c) { return c; }

_DH cfloat make_cuComplex(bool x)        { return ::make_cuComplex(x,0); }
_DH cfloat make_cuComplex(int x)         { return ::make_cuComplex(x,0); }
_DH cfloat make_cuComplex(unsigned x)    { return ::make_cuComplex(x,0); }
_DH cfloat make_cuComplex(float x)       { return ::make_cuComplex(x,0); }
_DH cfloat make_cuComplex(double x)      { return ::make_cuComplex(x,0); }
_DH cfloat make_cuComplex(cfloat x)   { return x; }
_DH cfloat make_cuComplex(cdouble c)  { return ::make_cuComplex(c.x,c.y); }

_DH cdouble make_cuDoubleComplex(bool x)        { return ::make_cuDoubleComplex(x,0); }
_DH cdouble make_cuDoubleComplex(int x)         { return ::make_cuDoubleComplex(x,0); }
_DH cdouble make_cuDoubleComplex(unsigned x)    { return ::make_cuDoubleComplex(x,0); }
_DH cdouble make_cuDoubleComplex(float x)       { return ::make_cuDoubleComplex(x,0); }
_DH cdouble make_cuDoubleComplex(double x)      { return ::make_cuDoubleComplex(x,0); }
_DH cdouble make_cuDoubleComplex(cdouble x)  { return x; }
_DH cdouble make_cuDoubleComplex(cfloat c)   { return ::make_cuDoubleComplex(c.x,c.y); }

///// complex * real
_DH cfloat operator*(cfloat a, float b)    { return ::make_cuComplex( real(a) * b, imag(a) * b); }
_DH cdouble operator*(cdouble a, float b)  { return ::make_cuDoubleComplex(real(a) * b, imag(a) * b); }
_DH cdouble operator*(cfloat a, double b)  { return ::make_cuDoubleComplex(real(a) * b, imag(a) * b); }
_DH cdouble operator*(cdouble a, double b) { return ::make_cuDoubleComplex(real(a) * b, imag(a) * b); }
_DH cfloat operator*(cfloat a, bool b)    { return a * (float)b; }
_DH cdouble operator*(cdouble a, bool b)    { return a * (double)b; }


_DH cfloat  operator*(float a, cfloat b)   { return b * a; }
_DH cdouble operator*(double a, cfloat b)  { return b * a; }
_DH cdouble operator*(float a, cdouble b)  { return b * a; }
_DH cdouble operator*(double a, cdouble b) { return b * a; }


///// complex * complex
_DH cfloat   operator*(cfloat a, cfloat b)   { return cuCmulf(a,b); }
_DH cdouble  operator*(cdouble a, cfloat b)  { return cuCmul(a,upcast(b)); }
_DH cdouble  operator*(cfloat a, cdouble b)  { return cuCmul(upcast(a),b); }
_DH cdouble  operator*(cdouble a, cdouble b) { return cuCmul(a,b); }


///// complex - complex
_DH cfloat   operator-(cfloat a, cfloat b)   { return cuCsubf(a,b); }
_DH cdouble  operator-(cdouble a, cfloat b)  { return cuCsub(a,upcast(b)); }
_DH cdouble  operator-(cfloat a, cdouble b)  { return cuCsub(upcast(a),b); }
_DH cdouble  operator-(cdouble a, cdouble b) { return cuCsub(a,b); }

//        DIV
// complex / complex
_DH static cfloat  operator/(cfloat  a, float b)   { return ::make_cuComplex( real(a) / b, imag(a) / (b)); }
_DH static cdouble operator/(cdouble a, float b)   { return ::make_cuDoubleComplex(real(a) / b, imag(a) / (b)); }

_DH cfloat min(cfloat a, cfloat b) { return cuCabsf(a) < cuCabsf(b) ? a : b; }
_DH cdouble min(cdouble a, cdouble b) { return cuCabs(a) < cuCabs(b) ? a : b; }
_DH cfloat min(float a, cfloat b) { return fabsf(a) < cuCabsf(b) ? make_cuComplex(a) : b; }
_DH cdouble min(double a, cdouble b) { return fabs(a) < cuCabs(b) ? make_cuDoubleComplex(a) : b; }

_DH cfloat max(cfloat a, cfloat b) { return cuCabsf(a) > cuCabsf(b) ? a : b; }
_DH cdouble max(cdouble a, cdouble b) { return cuCabs(a) > cuCabs(b) ? a : b; }
_DH cfloat max(float a, cfloat b) { return fabsf(a) > cuCabsf(b) ? make_cuComplex(a) : b; }
_DH cdouble max(double a, cdouble b) { return fabs(a) > cuCabs(b) ? make_cuDoubleComplex(a) : b; }

#define NEQ_OPS(OP, OT)                                     \
    _DH static bool operator OP(cfloat a, float b)       \
    { return (imag(a) OP 0 OT real(a) OP b); }              \
    _DH static bool operator OP(cdouble a, double b)     \
    { return (imag(a) OP 0 OT real(a) OP b); }              \
    _DH static bool operator OP(float b, cfloat a)       \
    { return (imag(a) OP 0 OT real(a) OP b); }              \
    _DH static bool operator OP(double b, cdouble a)     \
    { return (imag(a) OP 0 OT real(a) OP b); }              \
    _DH static bool operator OP(cfloat a, cfloat b)   \
    { return (imag(a) OP imag(b) OT real(a) OP real(b)); }  \
    _DH static bool operator OP(cdouble a, cdouble b) \
    { return (imag(a) OP imag(b) OT real(a) OP real(b)); }

NEQ_OPS(==, &&);
NEQ_OPS(!=, ||);

_DH bool operator*(bool a, cfloat b)    { return a & (b != 0); }
_DH bool operator*(bool a, cdouble b)    { return a & (b != 0); }
_DH bool operator+(bool a, cfloat b)    { return a | (b != 0); }
_DH bool operator+(bool a, cdouble b)    { return a | (b != 0); }

// from muxa_cplx.h
// complex *= complex
_DH static cfloat operator *=(cfloat &x, cfloat y){
    float re = cuCrealf(x)*cuCrealf(y) - cuCimagf(x)*cuCimagf(y);
    float im = cuCrealf(x)*cuCimagf(y) + cuCimagf(x)*cuCrealf(y);
    x.x = re;
    x.y = im;
    return x;
}

//        DIV
// complex / complex
_DH static cfloat operator/(cfloat a, cfloat b)   { return cuCdivf(a, b); }

// complex / cuCrealf (inv)
_DH static cfloat operator/(float a, cfloat b)        { return ::make_cuComplex(a, 0.0f) / b; }

// complex /= complex
_DH static cfloat operator /=(cfloat &x, cfloat y){
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
_DH static cfloat operator+(cfloat a, float b)        { return ::make_cuComplex(cuCrealf(a) + b, cuCimagf(a)); }
_DH static cfloat operator+(float a, cfloat b)        { return b + a; }

// HACK
_DH static cfloat operator+(cfloat a, bool b)    { return a; }
_DH static cdouble operator+(cdouble a, bool b)    { return a; }

// complex + complex
_DH static cfloat operator+(cfloat a, cfloat b)   { return cuCaddf(a, b); }

// complex += complex
_DH static cfloat operator +=(cfloat &x, cfloat y){
    x.x = cuCrealf(x) + cuCrealf(y);
    x.y = cuCimagf(x) + cuCimagf(y);
    return x;
}

// +complex
_DH static cfloat operator +(cfloat x)                { return x; }

//        SUB
// -complex
_DH static cfloat operator -(cfloat x)                 { return ::make_cuComplex(-cuCrealf(x), -cuCimagf(x)); }

// complex - cuCrealf (neg)
_DH static cfloat operator-(cfloat a, float b)         { return ::make_cuComplex(cuCrealf(a) - b, cuCimagf(a)); }
_DH static cfloat operator-(float a, cfloat b)         { return -b + a; }

// complex - complex
//_DH static cfloat operator-(cfloat a, cfloat b)    { return cuCsubf(a, b); }

// complex -= complex
_DH static cfloat operator -=(cfloat &x, cfloat y){
    x.x = cuCrealf(x) - cuCrealf(y);
    x.y = cuCimagf(x) - cuCimagf(y);
    return x;
}


//        MUL
// complex * complex
_DH static cdouble operator *=(cdouble &x, cdouble y){
    double re = cuCreal(x)*cuCreal(y) - cuCimag(x)*cuCimag(y);
    double im = cuCreal(x)*cuCimag(y) + cuCimag(x)*cuCreal(y);
    x.x = re;
    x.y = im;
    return x;
}
_DH static cdouble operator *=(cdouble &x, cfloat y) { return x *= upcast(y); }

//        DIV
// complex / complex
_DH static cdouble operator/(cdouble a, cdouble b)    { return cuCdiv(a, b); }
_DH static cdouble operator/(cdouble a, cfloat b)    { return a/upcast(b); }
_DH static cdouble operator/(cfloat a, cdouble b)    { return upcast(a)/b; }

// complex / cuCreal (inv)
_DH static cdouble operator/(cfloat a, double b)        { return upcast(a) / ::make_cuDoubleComplex(b, 0.0); }
_DH static cdouble operator/(cdouble a, double b)        { return a / ::make_cuDoubleComplex(b, 0.0); }
_DH static cdouble operator/(float a, cdouble b)         { return upcast(::make_cuComplex(a, 0.0f)) / b; }
_DH static cdouble operator/(double a, cfloat b)        { return ::make_cuDoubleComplex(a, 0.0) / upcast(b); }
_DH static cdouble operator/(double a, cdouble b)        { return ::make_cuDoubleComplex(a, 0.0) / b; }

// complex /= complex
_DH static cdouble operator /=(cdouble &x, cdouble y) {
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
_DH static cdouble operator /=(cdouble &x, cfloat y)  { return x /= upcast(y); }

//        ADD
// complex + cuCreal
_DH static cdouble operator+(cdouble a, float b)          { return ::make_cuDoubleComplex(cuCreal(a) + b, cuCimag(a)); }
_DH static cdouble operator+(cfloat a, double b)         { return ::make_cuDoubleComplex(cuCrealf(a) + b, cuCimagf(a)); }
_DH static cdouble operator+(cdouble a, double b)         { return ::make_cuDoubleComplex(cuCreal(a) + b, cuCimag(a)); }
_DH static cdouble operator+(float a, cdouble b)          { return b + a; }
_DH static cdouble operator+(double a, cfloat b)         { return b + a; }
_DH static cdouble operator+(double a, cdouble b)         { return b + a; }

// complex + complex
_DH static cdouble operator+(cdouble a, cfloat b)     { return cuCadd(a, upcast(b)); }
_DH static cdouble operator+(cfloat a, cdouble b)     { return cuCadd(upcast(a), b); }
_DH static cdouble operator+(cdouble a, cdouble b)     { return cuCadd(a, b); }

// complex += complex
_DH static cdouble operator +=(cdouble &x, cdouble y)  {
    x.x = cuCreal(x) + cuCreal(y);
    x.y = cuCimag(x) + cuCimag(y);
    return x;
}
_DH static cdouble operator +=(cdouble &x, cfloat y)  { return x += upcast(y); }
_DH static cfloat operator +=(cfloat &x, cdouble y)  { return x += downcast(y); }

// +complex
_DH static cdouble operator +(cdouble &x)                 { return x; }

//        SUB
// -complex
_DH static cdouble operator -(cdouble x)                  { return ::make_cuDoubleComplex(-cuCreal(x), -cuCimag(x)); }

// complex - cuCreal (neg)
_DH static cdouble operator-(cdouble a, float b)          { return ::make_cuDoubleComplex(cuCreal(a) - b, cuCimag(a)); }
_DH static cdouble operator-(cfloat a, double b)         { return ::make_cuDoubleComplex(cuCrealf(a) - b, cuCimagf(a)); }
_DH static cdouble operator-(cdouble a, double b)         { return ::make_cuDoubleComplex(cuCreal(a) - b, cuCimag(a)); }
_DH static cdouble operator-(float a, cdouble b)          { return -b + a; }
_DH static cdouble operator-(double a, cfloat b)         { return -b + a; }
_DH static cdouble operator-(double a, cdouble b)         { return -b + a; }

// complex -= complex
_DH static cdouble operator -=(cdouble &x, cdouble y)  {
    x.x = cuCreal(x) - cuCreal(y);
    x.y = cuCimag(x) - cuCimag(y);
    return x;
}
_DH static cdouble operator -=(cdouble &x, cfloat y)  { return x -= upcast(y); }

//        FUNCTIONS

_DH float abs(const cfloat x) { return cuCabsf(x); }
_DH double abs(const cdouble x){ return cuCabs(x); }

#define D_i            ::make_cuDoubleComplex(0.0, 1.0)
#define F_i            ::make_cuComplex(1.0, 0.0)


// FIXME: Remove this assert when 292 is fixed
_DH cdouble    sqrt(cdouble x){
    assert(1!=1);
    double rho = cuCabs(x);
    return x;  // Dummy
    //return sqrt((rho + cuCreal(x))/2.0) + D_i*(cuCimag(x)/(sqrt(2.0*(rho + cuCreal(x)))));
}

_DH cfloat    sqrt(cfloat x){
    float rho = cuCabsf(x);
    return sqrtf((rho + cuCrealf(x))/2.0f) + F_i*(cuCimagf(x)/(sqrtf(2.0f*(rho + cuCrealf(x)))));
}

static inline double cuCabs(cfloat x) { return cuCabsf(x); }
static inline double cuCreal(double x)   { return x; }
static inline double cuCimag(double x)   { return 0; }

} // namespace cuda
#undef _DH
