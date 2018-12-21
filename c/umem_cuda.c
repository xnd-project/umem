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
  switch(dest_ctx->type) {
  case umemHostDevice:
    umemCuda_copy_to_Host(src_ctx, (umemCuda * const)src_ctx, src_adr,  (umemHost * const)dest_ctx, dest_adr, nbytes);
    break;
  case umemCudaDevice:
    umemCuda_copy_to_Cuda(src_ctx, (umemCuda * const)src_ctx, src_adr, (umemCuda * const)dest_ctx, dest_adr, nbytes);
    break;
#ifdef HAVE_RMM_CONTEXT
  case umemRMMDevice:
    umemRMM_copy_from_Cuda(src_ctx, (umemRMM * const)dest_ctx, dest_adr, (umemCuda * const)src_ctx, src_adr, nbytes, false);
    break;
#endif
  default:
    umem_copy_to_via_host(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
  }
}

static void umemCuda_copy_from_(umemVirtual * const dest_ctx, uintptr_t dest_adr,
				umemVirtual * const src_ctx, uintptr_t src_adr,
				size_t nbytes) {
  assert(dest_ctx->type == umemCudaDevice);
  switch(src_ctx->type) {
  case umemHostDevice:
    umemCuda_copy_from_Host(dest_ctx, (umemCuda * const)dest_ctx, dest_adr,  (umemHost * const)src_ctx, src_adr, nbytes);
    break;
  case umemCudaDevice:
    umemCuda_copy_to_Cuda(dest_ctx, (umemCuda * const)dest_ctx, dest_adr, (umemCuda * const)src_ctx, src_adr, nbytes);
    break;
#ifdef HAVE_RMM_CONTEXT
  case umemRMMDevice:
    umemRMM_copy_to_Cuda(dest_ctx, (umemRMM * const)src_ctx, src_adr, (umemCuda * const)dest_ctx, dest_adr, nbytes, false);
    break;
#endif
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

