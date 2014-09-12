#include <err_common.hpp>
#include <type_util.hpp>
#include <string>
#include <iostream>
#include <sstream>

using std::string;
using std::stringstream;
using std::cerr;

AfError::AfError(const char * const funcName,
                 const int line,
                 const char * const message, af_err err)
    : logic_error   (message),
      functionName  (funcName),
      lineNumber(line),
      error(err)
{}

AfError::AfError(string funcName,
                 const int line,
                 string message, af_err err)
    : logic_error   (message),
      functionName  (funcName),
      lineNumber(line),
      error(err)
{}

const string&
AfError::getFunctionName() const
{
    return functionName;
}

int
AfError::getLine() const
{
    return lineNumber;
}

af_err
AfError::getError() const
{
    return error;
}

AfError::~AfError() throw() {}

TypeError::TypeError(const char * const  funcName,
                     const int line,
                     const int index, const af_dtype type)
    : AfError (funcName, line, "Invalid data type", AF_ERR_INVALID_TYPE),
      argIndex(index),
      errTypeName(getName(type))
{}

const string& TypeError::getTypeName() const
{
    return errTypeName;
}

const int TypeError::getArgIndex() const
{
	return argIndex;
}

DimensionError::DimensionError(const char * const  funcName,
                               const int line,
                               const int index,
                               const char * const  expectString)
    : AfError(funcName, line, "Invalid dimension", AF_ERR_SIZE),
      argIndex(index),
      expected(expectString)
{

}

const string& DimensionError::getExpectedCondition() const
{
	return expected;
}

const int DimensionError::getArgIndex() const
{
	return argIndex;
}


SupportError::SupportError(const char * const funcName,
                           const int line,
                           const char * const back)
    : AfError(funcName, line, "Unsupported Error", AF_ERR_NOT_SUPPORTED),
      backend(back)
{}

const string& SupportError::getBackendName() const
{
	return backend;
}


af_err processException()
{
    stringstream    ss;
    af_err          err= AF_ERR_INTERNAL;

    try {
        throw;
    } catch (const DimensionError &ex) {

        ss << "In function " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << "Invalid dimension for argument " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition() << "\n";

        cerr << ss.str();
        err = AF_ERR_SIZE;

    } catch (const SupportError &ex) {

        ss << ex.getFunctionName()
           << " not supported for " << ex.getBackendName()
           << " backend\n";

        cerr << ss.str();
        err = AF_ERR_NOT_SUPPORTED;
    } catch (const TypeError &ex) {

        ss << "In function " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << "Invalid type for argument " << ex.getArgIndex() << "\n";

        cerr << ss.str();
        err = AF_ERR_INVALID_TYPE;
    } catch (const AfError &ex) {

        ss << "Internal error in " << ex.getFunctionName()
           << "(" << ex.getLine() << "):\n"
           << ex.what() << "\n";

        cerr << ss.str();
        err = ex.getError();
    } catch (...) {

        cerr << "Unknown error\n";
        err = AF_ERR_UNKNOWN;
    }

    return err;
}
