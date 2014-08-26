#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <af/dim4.hpp>

typedef unsigned char uchar;

template<typename inType, typename outType, typename FileElementType>
void readTests(const std::string &FileName, std::vector<af::dim4> &inputDims,
                std::vector<std::vector<inType>>  &testInputs,
                std::vector<std::vector<outType>> &testOutputs)
{
    using std::vector;

    std::ifstream testFile(FileName);
    if(testFile.good()) {
        unsigned inputCount;
        testFile >> inputCount;
        for(unsigned i=0; i<inputCount; i++) {
            af::dim4 temp(1);
            testFile >> temp;
            inputDims.push_back(temp);
        }

        unsigned testCount;
        testFile >> testCount;
        testOutputs.resize(testCount);

        vector<unsigned> testSizes(testCount);
        for(unsigned i = 0; i < testCount; i++) {
            testFile >> testSizes[i];
        }

        testInputs.resize(inputCount,vector<inType>(0));
        for(unsigned k=0; k<inputCount; k++) {
            unsigned nElems = inputDims[k].elements();
            testInputs[k].resize(nElems);
            FileElementType tmp;
            for(unsigned i = 0; i < nElems; i++) {
                testFile >> tmp;
                testInputs[k][i] = tmp;
            }
        }

        testOutputs.resize(testCount, vector<outType>(0));
        for(unsigned i = 0; i < testCount; i++) {
            testOutputs[i].resize(testSizes[i]);
            FileElementType tmp;
            for(unsigned j = 0; j < testSizes[i]; j++) {
                testFile >> tmp;
                testOutputs[i][j] = tmp;
            }
        }

    }
    else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}
