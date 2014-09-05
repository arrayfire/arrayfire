#include <error.hpp>
#include <string>

using std::string;

AfError::AfError( const char * const message
            ,const char * const funcName)
        : logic_error   (message)
        , functionName  (funcName)
    {}
AfError::AfError( string message
                , string funcName)
        : logic_error   (message)
        , functionName  (funcName)
    {}

AfError::~AfError() throw() {}

TypeError::TypeError(
                    const char * const  funcName,
                    const char * const  argName,
                    af_dtype            expectedType,
                    af_dtype            actual)
: AfError (string("Invalid type[") + funcName + "]", funcName)
{}

ArgumentError::ArgumentError(
                    const char * const  funcName,
                    const char * const  argName,
                    int                 argIndex,
                    const char * const  actual)
: AfError (string("Invalid argument[") + funcName + "]:\nIn Function: " + funcName +
        + "\nArgument " + argName
        + "\nActual" + actual, funcName)
{}

DimensionError::DimensionError(
                    const char * const  funcName,
                    const char * const  expectString,
                    dim_type            actual)
: AfError(string("Invalid dimension[") + funcName
        + "]:\nExpected " + string(expectString), funcName)
, expected(expectString)
, value(actual)
{

}

SupportError::SupportError(const char * const funcName
        , const char * const back
        , const char * const condition)
    :   AfError(string("Unsupported Error[") + funcName + "]:" + string(condition) + " not supported by " + backend, funcName)
    ,   cond(condition)
    ,   backend(back)
    {}
