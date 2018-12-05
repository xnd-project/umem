#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include "umem.h"

#define CUDA_CALL(ME, CALL, ERROR, ERRRETURN, FMT, ...)			\
  do {									\
    int old_errno = errno;						\
    cudaError_t error = CALL;						\
    if (error != cudaSuccess) {						\
      char buf[256];							\
      snprintf(buf, sizeof(buf), FMT " -> %s: %s", __VA_ARGS__,		\
	       cudaGetErrorName(error), cudaGetErrorString(error));	\
      umem_set_status(ME, ERROR, buf);					\
      ERRRETURN;							\
    } else errno = old_errno;						\
  } while (0)

/*
  Implementations of umemCuda methods.
*/
static void umemCuda_dtor_(umemVirtual * const me) {
  umemVirtual_dtor(me);
}

static uintptr_t umemCuda_alloc_(umemVirtual * const me, size_t nbytes) {
  assert(me->type == umemCudaDevice);
  umemCuda * const me_ = (umemCuda * const)me;
  uintptr_t adr;
  CUDA_CALL(me, cudaMalloc((void**)&adr, nbytes), umemMemoryError, return 0,
	    "umemCuda_alloc_: cudaMalloc(&%" PRIxPTR ", %zu)",
	    adr, nbytes);
  return adr;
}

static void umemCuda_free_(umemVirtual * const me, uintptr_t adr) {
  assert(me->type == umemCudaDevice);
  umemCuda * const me_ = (umemCuda * const)me;
  CUDA_CALL(me, cudaFree((void*)adr), umemMemoryError, return,
	    "umemCuda_free_: cudaFree(%" PRIxPTR ")", adr); 
}

static void umemCuda_set_(umemVirtual * const me, uintptr_t adr, int c, size_t nbytes) {
  assert(me->type == umemCudaDevice);
  CUDA_CALL(me, cudaMemset((void*)adr, c, nbytes), umemMemoryError,return,
	    "umemCuda_set_: cudaMemset(&%" PRIxPTR ", %d, %zu)", adr, c, nbytes);
}


static void umemCuda_copy_to_(umemVirtual * const me, uintptr_t src_adr,
			      umemVirtual * const she, uintptr_t dest_adr,
			      size_t nbytes) {
  assert(me->type == umemCudaDevice);
  umemCuda * const me_ = (umemCuda * const)me;
  switch(she->type) {
  case umemHostDevice:
    {
      umemHost * const she_ = (umemHost * const)she;
      CUDA_CALL(me, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
			       nbytes, cudaMemcpyDeviceToHost), umemMemoryError, return,
		"umemCuda_copy_to_: cudaMemcpy(%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToHost)",
		dest_adr, src_adr, nbytes);
    }
    break;
  case umemCudaDevice:
    {
      umemCuda * const she_ = (umemCuda * const)she;
      if (me_->device == she_->device) {
	CUDA_CALL(me, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
				 nbytes, cudaMemcpyDeviceToDevice),
		  umemMemoryError, return,
		  "umemCuda_copy_to_: cudaMemcpy(%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToDevice)",
		  dest_adr, src_adr, nbytes);
      } else {
	CUDA_CALL(me, cudaMemcpyPeer((void*)dest_adr, she_->device,
				     (const void*)src_adr, me_->device,
				     nbytes), umemMemoryError, return,
		  "umemCuda_copy_to_: cudaMemcpyPeer(%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu)",
		  dest_adr, she_->device, src_adr, me_->device, nbytes);	
      }
    }
    break;
  default:
    umem_copy_to_via_host(me, src_adr, she, dest_adr, nbytes);
  }
}

static void umemCuda_copy_from_(umemVirtual * const me, uintptr_t dest_adr,
				umemVirtual * const she, uintptr_t src_adr,
				size_t nbytes) {
  assert(me->type == umemCudaDevice);
  umemCuda * const me_ = (umemCuda * const)me;
  switch(she->type) {
  case umemHostDevice:
    {
      umemHost * const she_ = (umemHost * const)she;
      CUDA_CALL(me, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
			       nbytes, cudaMemcpyHostToDevice),
		umemMemoryError, return,
		"umemCuda_copy_from_: cudaMemcpy(%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToDevice)",
		dest_adr, src_adr, nbytes);
    }
    break;
  case umemCudaDevice:
    umemCuda_copy_to_(she, dest_adr, me, src_adr, nbytes);
    break;
  default:
    umem_copy_from_via_host(me, dest_adr, she, src_adr, nbytes);
  }
}

/*
  umemCuda constructor.
*/

void umemCuda_ctor(umemCuda * const me, int device) {
  static struct umemVtbl const vtbl = {
    &umemCuda_dtor_,
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
  umemVirtual_ctor(&me->super);
  me->super.vptr = &vtbl;
  me->super.type = umemCudaDevice;
  me->device = device;

  int count;
  CUDA_CALL(&me->super, cudaGetDeviceCount(&count),
	    umemRuntimeError, return,
	    "umemCuda_ctor: cudaGetDeviceCount(&%d)",
	    count); 
  if (!(device >=0 && device < count)) {
    char buf[256];
    snprintf(buf, sizeof(buf),
	     "umemCuda_ctor: invalid device number: %d. Must be less than %d",
	     device, count);
    umem_set_status(&me->super, umemValueError, buf);
    return;
  }
  CUDA_CALL(&me->super, cudaSetDevice(device), umemRuntimeError, return,
	    "umemCuda_ctor: cudaSetDevice(%d)", device);
}


