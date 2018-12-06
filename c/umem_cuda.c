#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include "umem.h"

#define CUDA_CALL(THIS, CALL, ERROR, ERRRETURN, FMT, ...)			\
  do {									\
    int old_errno = errno;						\
    cudaError_t error = CALL;						\
    if (error != cudaSuccess) {						\
      char buf[256];							\
      snprintf(buf, sizeof(buf), FMT " -> %s: %s", __VA_ARGS__,		\
	       cudaGetErrorName(error), cudaGetErrorString(error));	\
      umem_set_status(THIS, ERROR, buf);					\
      ERRRETURN;							\
    } else errno = old_errno;						\
  } while (0)

/*
  Implementations of umemCuda methods.
*/
static void umemCuda_dtor_(umemVirtual * const this) {
  umemVirtual_dtor(this);
}

static uintptr_t umemCuda_alloc_(umemVirtual * const this, size_t nbytes) {
  assert(this->type == umemCudaDevice);
  umemCuda * const this_ = (umemCuda * const)this;
  uintptr_t adr;
  CUDA_CALL(this, cudaMalloc((void**)&adr, nbytes), umemMemoryError, return 0,
	    "umemCuda_alloc_: cudaMalloc(&%" PRIxPTR ", %zu)",
	    adr, nbytes);
  return adr;
}

static void umemCuda_free_(umemVirtual * const this, uintptr_t adr) {
  assert(this->type == umemCudaDevice);
  umemCuda * const this_ = (umemCuda * const)this;
  CUDA_CALL(this, cudaFree((void*)adr), umemMemoryError, return,
	    "umemCuda_free_: cudaFree(%" PRIxPTR ")", adr); 
}

static void umemCuda_set_(umemVirtual * const this, uintptr_t adr, int c, size_t nbytes) {
  assert(this->type == umemCudaDevice);
  CUDA_CALL(this, cudaMemset((void*)adr, c, nbytes), umemMemoryError,return,
	    "umemCuda_set_: cudaMemset(&%" PRIxPTR ", %d, %zu)", adr, c, nbytes);
}


static void umemCuda_copy_to_(umemVirtual * const this, uintptr_t src_adr,
			      umemVirtual * const that, uintptr_t dest_adr,
			      size_t nbytes) {
  assert(this->type == umemCudaDevice);
  umemCuda * const this_ = (umemCuda * const)this;
  switch(that->type) {
  case umemHostDevice:
    {
      umemHost * const that_ = (umemHost * const)that;
      CUDA_CALL(this, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
			       nbytes, cudaMemcpyDeviceToHost), umemMemoryError, return,
		"umemCuda_copy_to_: cudaMemcpy(%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToHost)",
		dest_adr, src_adr, nbytes);
    }
    break;
  case umemCudaDevice:
    {
      umemCuda * const that_ = (umemCuda * const)that;
      if (this_->device == that_->device) {
	CUDA_CALL(this, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
				 nbytes, cudaMemcpyDeviceToDevice),
		  umemMemoryError, return,
		  "umemCuda_copy_to_: cudaMemcpy(%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToDevice)",
		  dest_adr, src_adr, nbytes);
      } else {
	CUDA_CALL(this, cudaMemcpyPeer((void*)dest_adr, that_->device,
				     (const void*)src_adr, this_->device,
				     nbytes), umemMemoryError, return,
		  "umemCuda_copy_to_: cudaMemcpyPeer(%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu)",
		  dest_adr, that_->device, src_adr, this_->device, nbytes);	
      }
    }
    break;
  default:
    umem_copy_to_via_host(this, src_adr, that, dest_adr, nbytes);
  }
}

static void umemCuda_copy_from_(umemVirtual * const this, uintptr_t dest_adr,
				umemVirtual * const that, uintptr_t src_adr,
				size_t nbytes) {
  assert(this->type == umemCudaDevice);
  umemCuda * const this_ = (umemCuda * const)this;
  switch(that->type) {
  case umemHostDevice:
    {
      umemHost * const that_ = (umemHost * const)that;
      CUDA_CALL(this, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
			       nbytes, cudaMemcpyHostToDevice),
		umemMemoryError, return,
		"umemCuda_copy_from_: cudaMemcpy(%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToDevice)",
		dest_adr, src_adr, nbytes);
    }
    break;
  case umemCudaDevice:
    umemCuda_copy_to_(that, dest_adr, this, src_adr, nbytes);
    break;
  default:
    umem_copy_from_via_host(this, dest_adr, that, src_adr, nbytes);
  }
}

static bool umemCuda_is_same_device_(umemVirtual * const this, umemVirtual * const that) {
  umemCuda * const this_ = (umemCuda * const)this;
  umemCuda * const that_ = (umemCuda * const)that;
  return (this_->device == that_->device ? true : false);
}

/*
  umemCuda constructor.
*/

void umemCuda_ctor(umemCuda * const this, int device) {
  static struct umemVtbl const vtbl = {
    &umemCuda_dtor_,
    &umemCuda_is_same_device_,
    &umemCuda_alloc_,
    &umemVirtual_calloc,
    &umemCuda_free_,
    &umemVirtual_aligned_alloc,
    &umemVirtual_aligned_origin,
    &umemVirtual_aligned_free,
    &umemCuda_set_,
    &umemCuda_copy_to_,
    &umemCuda_copy_from_,
  };
  assert(sizeof(CUdeviceptr) == sizeof(uintptr_t));
  umemVirtual_ctor(&this->super);
  this->super.vptr = &vtbl;
  this->super.type = umemCudaDevice;
  this->device = device;

  int count;
  CUDA_CALL(&this->super, cudaGetDeviceCount(&count),
	    umemRuntimeError, return,
	    "umemCuda_ctor: cudaGetDeviceCount(&%d)",
	    count); 
  if (!(device >=0 && device < count)) {
    char buf[256];
    snprintf(buf, sizeof(buf),
	     "umemCuda_ctor: invalid device number: %d. Must be less than %d",
	     device, count);
    umem_set_status(&this->super, umemValueError, buf);
    return;
  }
  CUDA_CALL(&this->super, cudaSetDevice(device), umemRuntimeError, return,
	    "umemCuda_ctor: cudaSetDevice(%d)", device);
}


