#pragma once

#include <stdexcept>
#include <string>
#include <cassert>
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
	const std::string&
	getFunctionName() const;

    virtual ~AfError() throw();
};

class TypeError : public AfError
{
    TypeError();
public:
    TypeError(
                const char * const  funcName,
                const char * const  argName,
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
    std::string	expected;
public:
    DimensionError(
                    const char * const funcName,
                    const char * const expectString);
	const std::string&
	getExpectedCondition() const;

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

af_err processException();

#define DIM_ASSERT (COND, VAL)                               \
    if(COND == false) {                                     \
        #ifdef NDEBUG                                     	\
        throw DimensionError(__func__, "##COND##", VAL);    \
        #else                                               \
        assert(COND);                                       \
        #endif												\
    }

#define INVALID_TYPE_ERROR(type, argName, typeVal)          \
    throw TypeError(__func__, argName);

#define AFASSERT(COND, MESSAGE)                             \
    assert(MESSAGE && COND)

#define CATCHALL                		\
    catch(...) {                		\
        return processException();     	\
    }

