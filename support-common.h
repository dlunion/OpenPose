#pragma once

#ifdef __cplusplus
#define DllImport __declspec(dllimport)
#define DllExport __declspec(dllexport)
#else
#define DllImport
#define DllExport
#endif

#ifdef BuildDLL
#define Caffe_API DllExport
#else

#ifdef ImportDLL
#define Caffe_API DllImport
#else
#define Caffe_API 
#endif
#endif

#ifdef __cplusplus
template<typename Dtype>
class WPtr{
	typedef Dtype* DtypePtr;

	template<typename T>
	struct ptrInfo{
		T ptr;
		int refCount;

		ptrInfo(T p) :ptr(p), refCount(1){}
		void addRef(){ refCount++; }
		bool releaseRef(){ return --refCount <= 0; }
	};

public:
	WPtr() :ptr(0){};
	WPtr(DtypePtr p){
		ptr = new ptrInfo<DtypePtr>(p);
	}
	WPtr(const WPtr& other) :ptr(0){
		operator=(other);
	}
	~WPtr(){
		releaseRef();
	}

	void release(DtypePtr ptr);

	DtypePtr operator->(){
		return get();
	}

	operator DtypePtr(){
		return get();
	}

	WPtr& operator=(const WPtr& other){
		releaseRef();

		this->ptr = other.ptr;
		addRef();
		return *this;
	}

	DtypePtr get(){
		if (this->ptr)
			return ptr->ptr;
		return 0;
	}

	void addRef(){
		if (this->ptr)
			this->ptr->addRef();
	}

	void releaseRef(){
		if (this->ptr && this->ptr->releaseRef()){
			release(this->ptr->ptr);
			delete ptr;
			ptr = 0;
		}
	}

private:
	ptrInfo<DtypePtr>* ptr;
};

template<typename DtypePtr>
inline void WPtr<DtypePtr>::release(DtypePtr p){
	if (p) delete p;
}
#endif