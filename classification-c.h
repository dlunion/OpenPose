



#pragma once

#include "support-common.h"
#include <windows.h>

#define CC_Version			3.20
#define CC_Version_epoch    3
#define CC_Version_major    2
#define CC_Version_minor    0

#ifdef __cplusplus
#include <cv.h>
class Classifier;
typedef cv::Mat Image;
#else
typedef void Image;
typedef void Classifier;
#endif

#define BuildTaskPool

#ifdef __cplusplus
extern "C"{
#endif
	struct SoftmaxData{
		int label;
		float conf;
	};

	struct SoftmaxLayerOutput{
		int count;
		SoftmaxData* result;
	};

	struct SoftmaxResult{
		int count;
		SoftmaxLayerOutput* list;
	};

	struct MultiSoftmaxResult{
		int count;
		SoftmaxResult** list;
	};

	struct BlobData{
		int count;
		float* list;
		int num;
		int channels;
		int height;
		int width;
		int capacity_count;		//保留空间的元素个数长度，字节数请 * sizeof(float)
	};

#ifdef BuildTaskPool
	struct TaskPool{
		Classifier* model;
		int count_worker;

		volatile Image* recImgs;
		volatile int* top_n;
		volatile int* operType;
		volatile char** blobNames;

		volatile Image* cacheImgs;
		volatile int* cacheTop_n;
		volatile int* cacheOperType;
		volatile char** cacheBlobNames;

		volatile int recNum;
		volatile SoftmaxResult** recResult;
		volatile BlobData ** recBlobs;
		volatile int job_cursor;
		HANDLE semaphoreWait;
		volatile HANDLE* cacheSemaphoreGetResult;
		volatile HANDLE* semaphoreGetResult;
		//HANDLE semaphoreGetResult;
		HANDLE semaphoreGetResultFinish;
		CRITICAL_SECTION jobCS;
		volatile bool flag_run;
		volatile bool flag_exit;
		int gpu_id;
	};
#endif

	//setDecipherCallback
	const static int type_prototxt = 0;			//通过回调要求解密的部分是协议数据
	const static int type_caffemodel = 1;		//通过回调要求解密的部分是模型权重数据

	const static int event_decipher = 0;		//产生的事件要求解密
	const static int event_free = 1;			//产生的事件要求释放数据

	//event_decipher时回调的结果要求是数据指针，指针第一个4字节是数据长度，剩下是数据
	//event_free时，返回结果为0就行了，这时候length为0，data为event_decipher返回的数据指针
	typedef void* (__stdcall *DecipherCallback)(int event, int type, const void* data, int length);
	Caffe_API float __stdcall getVersion(DecipherCallback callback);
	Caffe_API float __stdcall getVersionEx(int* version_epoch, int* version_major, int* version_minor);

	Caffe_API void  __stdcall releaseBlobData(BlobData* ptr);
	Caffe_API void  __stdcall releaseSoftmaxResult(SoftmaxResult* ptr);
	Caffe_API void __stdcall releaseMultiSoftmaxResult(MultiSoftmaxResult* ptr);

	Caffe_API Classifier* __stdcall createClassifier(
		const char* prototxt_file,
		const char* caffemodel_file,
		float scale_raw = 1,
		const char* mean_file = 0,
		int num_means = 0,
		float* means = 0,
		int gpu_id = -1);

	Caffe_API Classifier* __stdcall createClassifierByData(
		const void* prototxt_data,
		int prototxt_data_length,
		const void* caffemodel_data,
		int caffemodel_data_length,
		float scale_raw = 1,
		const char* mean_file = 0,
		int num_means = 0,
		float* means = 0,
		int gpu_id = -1);

	Caffe_API void __stdcall releaseClassifier(Classifier* classifier);
	Caffe_API SoftmaxResult* __stdcall predictSoftmax(Classifier* classifier, const void* img, int len, int top_n = 5);
	Caffe_API MultiSoftmaxResult* __stdcall predictMultiSoftmax(Classifier* classifier, const void** img, int* len, int num, int top_n = 5);
	Caffe_API BlobData* __stdcall extfeature(Classifier* classifier, const void* img, int len, const char* feature_name);

	Caffe_API SoftmaxResult* __stdcall predictSoftmaxAny(Classifier* classifier, const float* data, const int* dims, int top_n = 5);
	Caffe_API MultiSoftmaxResult* __stdcall predictMultiSoftmaxAny(Classifier* classifier, const float** data, const int* dims, int num, int top_n = 5);
	Caffe_API BlobData* __stdcall extfeatureAny(Classifier* classifier, const float* data, const int* dims, const char* feature_name);

	//获取任意层的blob
	Caffe_API void __stdcall forward(Classifier* classifier, const void* img, int len);

	//调整输入尺寸
	Caffe_API void __stdcall reshape(Classifier* classifier, int width, int height);

	//裁图，保证缓存区的长度要够，outLen为指定的缓冲区长度，如果outLen不够长，则会返回false，只要够基本不会出现false
	Caffe_API bool __stdcall cropImage(const char* img, int len, bool color, int x, int y, int width, int height, char* buf, int* outlen, const char* ext);

	//裁图，保证缓存区的长度要够，outLen为指定的缓冲区长度，如果outLen不够长，则会返回false，只要够基本不会出现false
	Caffe_API bool __stdcall cropCenterImage(const char* img, int len, bool color, char* buf, int* outlen, const char* ext);

	//获取任意层的blob
	Caffe_API BlobData* __stdcall getBlobData(Classifier* classifier, const char* blob_name);

	//获取特征的长度
	Caffe_API int __stdcall getBlobLength(BlobData* feature);

	//获取张量数据维度
	Caffe_API void __stdcall getBlobDims(BlobData* blob, int* dims_at_4_elem);

	//将特征复制到缓存区
	Caffe_API void __stdcall cpyBlobData(void* buffer, BlobData* feature);

	//获取输出层的个数
	Caffe_API int __stdcall getNumOutlayers(SoftmaxResult* result);

	//获取层里面的数据个数
	Caffe_API int __stdcall getLayerNumData(SoftmaxLayerOutput* layer);

	//获取结果的label
	Caffe_API int __stdcall getResultLabel(SoftmaxResult* result, int layer, int num);

	//获取结果的置信度
	Caffe_API float __stdcall getResultConf(SoftmaxResult* result, int layer, int num);

	//获取里面的个数
	Caffe_API int __stdcall getMultiSoftmaxNum(MultiSoftmaxResult* multi);

	//获取里面元素的指针
	Caffe_API SoftmaxResult* __stdcall getMultiSoftmaxElement(MultiSoftmaxResult* multi, int index);

	//多标签就是多个输出层，每个层取softmax，注意buf的个数是getNumOutlayers得到的数目一致
	Caffe_API void __stdcall getMultiLabel(SoftmaxResult* result, int* buf);
	Caffe_API void __stdcall getMultiConf(SoftmaxResult* result, float* buf);

	//获取第0个输出的label
	Caffe_API int __stdcall getSingleLabel(SoftmaxResult* result);

	//获取第0个输出的置信度
	Caffe_API float __stdcall getSingleConf(SoftmaxResult* result);

	//获取最后发生的错误，没有错误返回0
	Caffe_API const char* __stdcall getLastErrorMessage();

	Caffe_API void __stdcall enablePrintErrorToConsole();

	Caffe_API void __stdcall disableErrorOutput();

#ifdef BuildTaskPool
	Caffe_API TaskPool* __stdcall createTaskPool(
		const char* prototxt_file,
		const char* caffemodel_file,
		float scale_raw = 1,
		const char* mean_file = 0,
		int num_means = 0,
		float* means = 0,
		int gpu_id = -1,
		int batch_size = 3);

	Caffe_API TaskPool* __stdcall createTaskPoolByData(
		const void* prototxt_data,
		int prototxt_data_length,
		const void* caffemodel_data,
		int caffemodel_data_length,
		float scale_raw = 1,
		const char* mean_file = 0,
		int num_means = 0,
		float* means = 0,
		int gpu_id = -1,
		int batch_size = 3);

	Caffe_API void __stdcall releaseTaskPool(TaskPool* taskPool);

	Caffe_API SoftmaxResult* __stdcall predictSoftmaxByTaskPool(TaskPool* pool, const void* img, int len, int top_n = 1);
	Caffe_API SoftmaxResult* __stdcall predictSoftmaxAnyByTaskPool(TaskPool* pool, const float* data, const int* dims, int top_n = 1);
	Caffe_API SoftmaxResult* __stdcall predictSoftmaxByTaskPool2(TaskPool* pool, const Image* img, int top_n = 1);
	Caffe_API BlobData* __stdcall forwardByTaskPool(TaskPool* pool, const void* img, int len, const char* blob_name);
	Caffe_API BlobData* __stdcall forwardByTaskPool2(TaskPool* pool, const Image* img, const char* blob_name);
#endif
#ifdef __cplusplus 
}; 
#endif
