
#include <stdexcept>
#include <string>
#include <af/defines.h>


class AfError   : public std::logic_error
{
    std::string functionName;
    AfError();
public:
    AfError( const char * const message
            ,const char * const funcName);
    AfError( std::string message
            ,std::string funcName);

    virtual ~AfError() throw();
};

class TypeError : public AfError
{
    TypeError();
public:
    TypeError(
                const char * const  funcName,
                const char * const  argName,
                af_dtype            expectedType,
                af_dtype            actual);
    ~TypeError() throw() {}
};

class ArgumentError : public AfError
{
    ArgumentError();
public:
    ArgumentError(
                    const char * const  funcName,
                    const char * const  argName,
                    int                 argIndex,
                    const char * const  actual);
    ~ArgumentError() throw(){}
};

class DimensionError : public AfError
{
    DimensionError();
    std::string      expected;
    dim_type    value;
public:
    DimensionError(
                    const char * funcName,
                    const char * expectString,
                    dim_type     actual);
    ~DimensionError() throw(){}
};

class SupportError  :   public AfError
{
    std::string cond;
    std::string backend;
    SupportError();
public:
    SupportError(const char * const funcName
            , const char * const back
            , const char * const condition);
    ~SupportError()throw() {}
};

#define DIM_CHECK(COND, VAL)                        \
    throw DimensionError(__func__, "##COND##", VAL);
