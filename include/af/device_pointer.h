#include<arrayfire.h>
#include<cuda_runtime.h>

namespace af
{
	template<typename Type>
	class device_pointer {

		af::array* parent;

	public:

		int cols;
		int rows;

		Type* data;

		device_pointer(af::array* _parent, dim4 _dims, Type* _data) {
			parent = _parent;
			cols = _dims[0];
			rows = _dims[1];
			data = _data;
		}

		__device__ Type& operator ()(int col, int row = 0) { return data[(col * rows) + row]; }

		//parent array unlocks once the device_pointer falls out of scope
		~device_pointer() {
			parent->unlock();
		}

	};

}