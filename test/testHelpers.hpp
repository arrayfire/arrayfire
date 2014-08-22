#include <string>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>

using std::copy;
using std::copy_n;
using std::string;
using std::vector;
using std::istream_iterator;
using std::ostream_iterator;

template<typename FileDataElementType, typename ArrayElementType>
void ReadTests(const string &FileName, af::dim4 &dims,
        vector<ArrayElementType> &testInput,
        vector<vector<ArrayElementType>> &testOutputs)
{
    std::ifstream testFile(FileName);
    if(testFile.good()) {
        testFile >> dims;
        vector<FileDataElementType> data(dims.elements());

        unsigned testCount;
        testFile >> testCount;
        testOutputs.resize(testCount);

        vector<unsigned> testSizes(testCount);
        for(unsigned i = 0; i < testCount; i++) {
            testFile >> testSizes[i];
        }

        copy_n( istream_iterator<FileDataElementType>(testFile),
                dims.elements(),
                begin(data));

        copy(   begin(data),
                end(data),
                back_inserter(testInput));

        for(unsigned i = 0; i < testCount; i++) {
            copy_n( istream_iterator<int>(testFile),
                    testSizes[i],
                    back_inserter(testOutputs[i]));
        }
    }
    else {
        FAIL() << "TEST FILE NOT FOUND";
    }
}
