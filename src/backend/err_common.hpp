#pragma once

#include <stdexcept>
#include <string>
#include <cassert>
#include <af/defines.h>
#include <vector>

class AfError   : public std::logic_error
{
    std::string functionName;
    AfError();

public:

    AfError( const char * const message
            ,const char * const funcName);

    AfError( std::string message
            ,std::string funcName);

	const std::string&
	getFunctionName() const;

    virtual ~AfError() throw();
};

// TODO: Perhaps add a way to return supported types
class TypeError : public AfError
{
    int argIndex;
    std::string errTypeName;
    TypeError();

public:

    TypeError(const char * const  funcName,
              const int index, const af_dtype type);

	const std::string&
	getTypeName() const;

    const int getArgIndex() const;

    ~TypeError() throw() {}
};

class DimensionError : public AfError
{
    int argIndex;
    std::string	expected;
    DimensionError();

public:
    DimensionError(const char * const funcName,
                   const int index,
                   const char * const expectString);

	const std::string&
	getExpectedCondition() const;

    const int getArgIndex() const;

    ~DimensionError() throw(){}
};

class SupportError  :   public AfError
{
    std::string backend;
    SupportError();
public:
    SupportError(const char * const funcName,
                 const char * const back);
    ~SupportError()throw() {}
	const std::string&
	getBackendName() const;
};

af_err processException();

#define DIM_ASSERT (INDEX, COND) do {                       \
        if(COND == false) {                                 \
            throw DimensionError(__func__, INDEX, #COND);   \
        }                                                   \
    } while(0)

#define TYPE_ERROR(INDEX, type) do {            \
        throw TypeError(__func__, INDEX, type); \
    } while(0)                                  \


#define AF_ASSERT(COND, MESSAGE)                \
    assert(MESSAGE && COND)

#define CATCHALL                                \
    catch(...) {                                \
        return processException();              \
    }
