
#include <stdexcept>
#include <string>

class AfError   : public std::logic_error
{
    string functionName;
    AfError();
public:
    AfError( const char * const message
            ,const char * const funcName)
        : logic_error   (message)
        , functionName  (funcName)
    {}

    virtual ~AfError() {}

};

class TypeError : public AfError
{
    TypeError();
public:
    explicit TypeError(
                        const char * const  funcName,
                        const char * const  argName,
                        af_dtype            expectedType)
    : AfError (string("Invalid type:\nIn Function: " + funcName +
            + "\nExpected " + expectedType
            + "\nActual" + to_string(actual)), funcName),
    {}
};

class ArgumentError : public AfError
{
    ArgumentError();
public:
    explicit arg_error(
                        const char * const  funcName,
                        const char * const  argName,
                        int                 argIndex)
    : AfError (string("Invalid argument:\nIn Function: " + funcName +
            + "\nExpected " + expectString
            + "\nActual" + to_string(actual)), funcName)
    {}
};

class DimensionError : public AfError
{
    DimensionError();
    string      expected;
    dim_size    value;
public:
    explicit DimensionError(
                        const char * const  funcName,
                        const char * const  expectString,
                        dim_size            &actual)
    : AfError(string("Invalid dimension:\nExpected ") + expectString + "\nActual", to_string(actual))
    , functionName(funcName)
    , expected(expectString)
    , value(actual)
    {

    }
};

class SupportError  :   public AfError
{
    string cond;
    string backend;
    SupportError();
public:
    SupportError(const char * const funcName
            , const char * const back
            , const char * const condition)
        :   AfError(string(condition) + " not supported by " + backend, funcName)
        ,   cond(condition)
        ,   backend(back)
        {}
};
