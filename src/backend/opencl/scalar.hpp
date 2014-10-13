#include <Array.hpp>
#include <optypes.hpp>
#include <math.hpp>
#include <JIT/ScalarNode.hpp>

namespace opencl
{

template<typename T> double upCast(T in) { return (double)(in); }
cdouble upCast(cfloat in) { cdouble out = {in.s[0], in.s[1]}; return out;}
cdouble upCast(cdouble in) { return in; }

template<typename T> static bool isDouble() { return false; }
template<> bool isDouble<double>() { return true; }
template<> bool isDouble<cdouble>() { return true; }

template<typename T>
Array<T>* createScalarNode(const dim4 &size, const T val)
{
    JIT::ScalarNode *node = NULL;
    node = new JIT::ScalarNode(upCast(val), isDouble<T>());
    return createNodeArray<T>(size, reinterpret_cast<JIT::Node *>(node));
}

}
