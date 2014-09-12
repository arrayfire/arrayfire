#include <err_common.hpp>
#include <type_util.hpp>
#include <string>
#include <iostream>
#include <sstream>

using std::string;
using std::stringstream;
using std::cerr;

AfError::AfError(const char * const message,
                 const char * const funcName)
    : logic_error   (message),
      functionName  (funcName)
{}

AfError::AfError(string message,
                 string funcName)
    : logic_error   (message),
      functionName  (funcName)
{}

const string&
AfError::getFunctionName() const
{
    return functionName;
}

AfError::~AfError() throw() {}

TypeError::TypeError(const char * const  funcName,
                     const int index, const af_dtype type)
    : AfError ("Invalid data type", funcName),
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
                               const int index,
                               const char * const  expectString)
    : AfError("Invalid dimension", funcName),
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
                           const char * const back)
    : AfError("Unsupported Error", funcName),
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

        ss << "In function " << ex.getFunctionName() << "\n"
           << "Invalid dimension for argument " << ex.getArgIndex() << "\n"
           << "Expected: " << ex.getExpectedCondition();

        cerr << ss.str();
        err = AF_ERR_SIZE;

    } catch (const SupportError &ex) {

        ss << ex.getFunctionName()
           << " not supported for " << ex.getBackendName()
           << " backend";

        cerr << ss.str();
        err = AF_ERR_NOT_SUPPORTED;
    } catch (const TypeError &ex) {

        ss << "In function " << ex.getFunctionName() << "\n"
           << "Invalid type for argument " << ex.getArgIndex() << "\n";

        cerr << ss.str();
        err = AF_ERR_INVALID_TYPE;
    } catch (const AfError &ex) {

        ss << "Internal error in " << ex.getFunctionName() << "\n";

        cerr << ss.str();
        err = AF_ERR_INTERNAL;
    } catch (...) {

        cerr << "Unknown error\n";
        err = AF_ERR_UNKNOWN;
    }

    return err;
}
