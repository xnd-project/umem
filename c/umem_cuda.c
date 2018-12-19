#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include "umem.h"

#include "umem_cuda_utils.h"

/*
  Implementations of umemCuda methods.
*/
static void umemCuda_dtor_(umemVirtual * const ctx) {
  umemVirtual_dtor(ctx);
}

static uintptr_t umemCuda_alloc_(umemVirtual * const ctx, size_t nbytes) {
  assert(ctx->type == umemCudaDevice);
  //umemCuda * const ctx_ = (umemCuda * const)ctx;
  uintptr_t adr;
  CUDA_CALL(ctx, cudaMalloc((void**)&adr, nbytes), umemMemoryError, return 0,
	    "umemCuda_alloc_: cudaMalloc(&%" PRIxPTR ", %zu)",
	    adr, nbytes);
  return adr;
}

static void umemCuda_free_(umemVirtual * const ctx, uintptr_t adr) {
  assert(ctx->type == umemCudaDevice);
  //umemCuda * const ctx_ = (umemCuda * const)ctx;
  CUDA_CALL(ctx, cudaFree((void*)adr), umemMemoryError, return,
	    "umemCuda_free_: cudaFree(%" PRIxPTR ")", adr); 
}

static void umemCuda_set_(umemVirtual * const ctx, uintptr_t adr, int c, size_t nbytes) {
  assert(ctx->type == umemCudaDevice);
  CUDA_CALL(ctx, cudaMemset((void*)adr, c, nbytes), umemMemoryError,return,
	    "umemCuda_set_: cudaMemset(&%" PRIxPTR ", %d, %zu)", adr, c, nbytes);
}


static void umemCuda_copy_to_(umemVirtual * const src_ctx, uintptr_t src_adr,
			      umemVirtual * const dest_ctx, uintptr_t dest_adr,
			      size_t nbytes) {
  assert(src_ctx->type == umemCudaDevice);
  umemCuda * const src_ctx_ = (umemCuda * const)src_ctx;
  switch(dest_ctx->type) {
  case umemHostDevice:
    {
      //umemHost * const dest_ctx_ = (umemHost * const)dest_ctx;
      CUDA_CALL(src_ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
			       nbytes, cudaMemcpyDeviceToHost), umemMemoryError, return,
		"umemCuda_copy_to_: cudaMemcpy(%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToHost)",
		dest_adr, src_adr, nbytes);
    }
    break;
  case umemCudaDevice:
    {
      umemCuda * const dest_ctx_ = (umemCuda * const)dest_ctx;
      if (src_ctx_->device == dest_ctx_->device) {
	CUDA_CALL(src_ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
				 nbytes, cudaMemcpyDeviceToDevice),
		  umemMemoryError, return,
		  "umemCuda_copy_to_: cudaMemcpy(%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyDeviceToDevice)",
		  dest_adr, src_adr, nbytes);
      } else {
	CUDA_CALL(src_ctx, cudaMemcpyPeer((void*)dest_adr, dest_ctx_->device,
				     (const void*)src_adr, src_ctx_->device,
				     nbytes), umemMemoryError, return,
		  "umemCuda_copy_to_: cudaMemcpyPeer(%" PRIxPTR ", %d, %" PRIxPTR ", %d, %zu)",
		  dest_adr, dest_ctx_->device, src_adr, src_ctx_->device, nbytes);	
      }
    }
    break;
  default:
    umem_copy_to_via_host(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
  }
}

static void umemCuda_copy_from_(umemVirtual * const dest_ctx, uintptr_t dest_adr,
				umemVirtual * const src_ctx, uintptr_t src_adr,
				size_t nbytes) {
  assert(dest_ctx->type == umemCudaDevice);
  //umemCuda * const dest_ctx_ = (umemCuda * const)dest_ctx;
  switch(src_ctx->type) {
  case umemHostDevice:
    {
      //umemHost * const src_ctx_ = (umemHost * const)src_ctx;
      CUDA_CALL(dest_ctx, cudaMemcpy((void*)dest_adr, (const void*)src_adr,
			       nbytes, cudaMemcpyHostToDevice),
		umemMemoryError, return,
		"umemCuda_copy_from_: cudaMemcpy(%" PRIxPTR ", %" PRIxPTR ", %zu, cudaMemcpyHostToDevice)",
		dest_adr, src_adr, nbytes);
    }
    break;
  case umemCudaDevice:
    umemCuda_copy_to_(src_ctx, dest_adr, dest_ctx, src_adr, nbytes);
    break;
  default:
    umem_copy_from_via_host(dest_ctx, dest_adr, src_ctx, src_adr, nbytes);
  }
}

static bool umemCuda_is_same_context_(umemVirtual * const one_ctx, umemVirtual * const other_ctx) {
  umemCuda * const one_ctx_ = (umemCuda * const)one_ctx;
  umemCuda * const other_ctx_ = (umemCuda * const)other_ctx;
  return (one_ctx_->device == other_ctx_->device ? true : false);
}

/*
  umemCuda constructor.
*/
void umemCuda_ctor(umemCuda * const ctx, int device) {
  static struct umemVtbl const vtbl = {
    &umemCuda_dtor_,
    &umemCuda_is_same_context_,
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
  umemHost_ctor(&ctx->host);
  umemVirtual_ctor(&ctx->super, &ctx->host);
  ctx->super.vptr = &vtbl;
  ctx->super.type = umemCudaDevice;
  ctx->device = device;

  int count;
  CUDA_CALL(&ctx->super, cudaGetDeviceCount(&count),
	    umemRuntimeError, return,
	    "umemCuda_ctor: cudaGetDeviceCount(&%d)",
	    count); 
  if (!(device >=0 && device < count)) {
    char buf[256];
    snprintf(buf, sizeof(buf),
	     "umemCuda_ctor: invalid device number: %d. Must be less than %d",
	     device, count);
    umem_set_status(&ctx->super, umemValueError, buf);
    return;
  }
  CUDA_CALL(&ctx->super, cudaSetDevice(device), umemRuntimeError, return,
	    "umemCuda_ctor: cudaSetDevice(%d)", device);
}

