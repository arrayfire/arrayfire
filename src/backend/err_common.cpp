#include <err_common.hpp>
#include <string>
#include <iostream>
#include <sstream>

using std::string;
using std::stringstream;
using std::cerr;

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

const string&
AfError::getFunctionName() const
{
    return functionName;
}

AfError::~AfError() throw() {}

TypeError::TypeError(     const char * const  funcName,
                        const char * const  argName,
                        af_dtype            actual)
        : AfError ("Invalid type", funcName)
{}

ArgumentError::ArgumentError(     const char * const  funcName,
                                const char * const  argName,
                                int                 argIndex,
                                const char * const  actual)
        : AfError ("Invalid argument", funcName)
{}

DimensionError::DimensionError( const char * const  funcName,
                                const char * const  expectString)
        : AfError("Invalid dimension", funcName)
        , expected(expectString)
{

}

const string&
DimensionError::getExpectedCondition() const
{
	return expected;
}

SupportError::SupportError( const char * const funcName
                            , const char * const back
                            , const char * const condition)
        :   AfError("Unsupported Error", funcName)
        ,   cond(condition)
        ,   backend(back)
    {}

af_err processException()
{
    stringstream    ss;
    af_err          err= AF_ERR_INTERNAL;
    try {
        throw;
    }
    catch (const DimensionError &ex) {
        ss  << "Invalid dimension[" << ex.getFunctionName() << "]:\n"
            << "Expected: " << ex.getExpectedCondition();
        cerr << ss.str();
        err = AF_ERR_SIZE;
    }
	// TODO: something along these lines
    //catch (const SupportError &ex) {
    //    ss  <<  "Unsupported Error[" << ex.getFunctionName() << "]:\n"
    //        <<  condition <<" not supported by " + backend;
    //    err = AF_ERR_NOT_SUPPORTED;

    //}
    //catch (const ArgumentError &ex) {
    //    ss  <<  "Invalid argument[" << ex.getFunctionName() << "]:\n"
    //        <<  "Argument " << argName
    //        <<  "Actual" << actual;
    //    err = AF_ERR_ARG;
    //}
    //catch (const TypeError &ex) {
    //    ss  <<  "Unsupported["   << ex.getFunctionName() << "]:\n"
    //        <<  condition;
    //    err = AF_ERR_INVALID_TYPE;
    //}
    return err;
}
