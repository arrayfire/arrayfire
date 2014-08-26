#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <af/dim4.hpp>

typedef unsigned char uchar;

template<typename FileDataElementType, typename ArrayElementType>
void ReadTests(const std::string &FileName, af::dim4 &dims,
        std::vector<ArrayElementType> &testInput,
        std::vector<std::vector<ArrayElementType>> &testOutputs)
{
    using std::copy;
    using std::string;
    using std::vector;

    std::ifstream testFile(FileName.c_str());
    if(testFile.good()) {
        testFile >> dims;

        unsigned testCount;
        testFile >> testCount;

        vector<unsigned> testSizes(testCount);
        for(unsigned i = 0; i < testCount; i++) {
            testFile >> testSizes[i];
        }

        vector<FileDataElementType> data(dims.elements());
        testInput.reserve(dims.elements());
        for(unsigned i = 0; i < dims.elements(); i++) {
            testFile >> data[i];
            testInput.push_back(data[i]); //convert to ArrayElementType
        }

        testOutputs.resize(testCount, vector<ArrayElementType>(0));
        for(unsigned i = 0; i < testCount; i++) {
            testOutputs[i].resize(testSizes[i]);
            FileDataElementType tmp;
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

template<typename inType, typename outType, typename FileElementType>
void ReadTests2(const string &FileName, af::dim4 &dims,
        vector<inType> &testInput,
        vector<vector<outType>> &testOutputs)
{
    std::ifstream testFile(FileName);
    if(testFile.good()) {
        testFile >> dims;
        vector<inType> data(dims.elements());

        unsigned testCount;
        testFile >> testCount;
        testOutputs.resize(testCount);

        vector<unsigned> testSizes(testCount);
        for(unsigned i = 0; i < testCount; i++) {
            testFile >> testSizes[i];
        }

        copy_n( istream_iterator<FileElementType>(testFile),
                dims.elements(),
                begin(data));

        copy(   begin(data),
                end(data),
                back_inserter(testInput));

        for(unsigned i = 0; i < testCount; i++) {
            copy_n( istream_iterator<outType>(testFile),
                    testSizes[i],
                    back_inserter(testOutputs[i]));
        }
    }
    else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}
