#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include "umem.h"
#include "umem_cuda_utils.h"


static void umemCudaManaged_dtor_(umemVirtual * const ctx) {
  umemVirtual_dtor(ctx);
}

static uintptr_t umemCudaManaged_alloc_(umemVirtual * const ctx, size_t nbytes) {
  assert(ctx->type == umemCudaDevice);
  umemCudaManaged * const ctx_ = (umemCudaManaged * const)ctx;
  uintptr_t adr;
  CUDA_CALL(ctx, cudaMallocManaged((void**)&adr, nbytes, ctx_->flags), umemMemoryError, return 0,
	    "umemCudaManaged_alloc_: cudaMallocManaged(&%" PRIxPTR ", %zu, %u)",
	    adr, nbytes, ctx_->flags);
  return adr;
}

static void umemCudaManaged_free_(umemVirtual * const ctx, uintptr_t adr) {
  assert(ctx->type == umemCudaManagedDevice);
  //umemCuda * const ctx_ = (umemCuda * const)ctx;
  CUDA_CALL(ctx, cudaFree((void*)adr), umemMemoryError, return,
	    "umemCudaManaged_free_: cudaFree(%" PRIxPTR ")", adr); 
}

static void umemCudaManaged_set_(umemVirtual * const ctx, uintptr_t adr, int c, size_t nbytes) {
  assert(ctx->type == umemCudaManagedDevice);
  umemCudaManaged * const ctx_ = (umemCudaManaged * const)ctx;
  umemCudaSet(ctx, adr, c, nbytes, ctx_->async, ctx_->stream);
}


static void umemCudaManaged_copy_to_(umemVirtual * const src_ctx, uintptr_t src_adr,
                                     umemVirtual * const dest_ctx, uintptr_t dest_adr,
                                     size_t nbytes) {
  assert(src_ctx->type == umemCudaDevice);
  umemCudaManaged * const src_ctx_ = (umemCudaManaged * const)src_ctx;
  switch(dest_ctx->type) {
  case umemHostDevice:
  case umemCudaManagedDevice:
    umemCudaCopyToHost(src_ctx, src_adr, dest_adr, nbytes, src_ctx_->async, src_ctx_->stream);
    break;
#ifdef HAVE_CUDA_CONTEXT
  case umemCudaDevice:
#endif
#ifdef HAVE_RMM_CONTEXT
  case umemRMMDevice:
#endif
    umemCudaCopyFromHost(src_ctx, dest_adr, src_adr, nbytes, src_ctx_->async, src_ctx_->stream);
    break;
  default:
    umem_copy_to_via_host(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
  }
}

static void umemCudaManaged_copy_from_(umemVirtual * const dest_ctx, uintptr_t dest_adr,
                                       umemVirtual * const src_ctx, uintptr_t src_adr,
                                       size_t nbytes) {
  assert(dest_ctx->type == umemCudaManagedDevice);
  umemCudaManaged * const dest_ctx_ = (umemCudaManaged * const)dest_ctx;
  switch(src_ctx->type) {
  case umemHostDevice:
  case umemCudaManagedDevice:
    umemCudaCopyFromHost(dest_ctx, dest_adr, src_adr, nbytes, false, 0);
    break;
#ifdef HAVE_CUDA_CONTEXT
  case umemCudaDevice:
#endif
#ifdef HAVE_RMM_CONTEXT
  case umemRMMDevice:
#endif
    umemCudaCopyToHost(dest_ctx, src_adr, dest_adr, nbytes, dest_ctx_->async, dest_ctx_->stream);
    break;
  default:
    umem_copy_from_via_host(dest_ctx, dest_adr, src_ctx, src_adr, nbytes);
  }
}

bool umemCudaManaged_is_accessible_from_(umemVirtual * const src_ctx, umemVirtual * const dest_ctx) {
  assert(src_ctx->type == umemCudaManagedDevice);
  switch (dest_ctx->type) {
  case umemHostDevice:
  case umemCudaManagedDevice:
    return true;
  default: ;
  }
  return false;
}

/*
  umemCudaManaged constructor.
*/
void umemCudaManaged_ctor(umemCudaManaged * const ctx, unsigned int flags, bool async, uintptr_t stream) {
  static struct umemVtbl const vtbl = {
    &umemCudaManaged_dtor_,
    &umemCudaManaged_is_accessible_from_,
    &umemCudaManaged_alloc_,
    &umemVirtual_calloc,
    &umemCudaManaged_free_,
    &umemVirtual_aligned_alloc,
    &umemVirtual_aligned_origin,
    &umemVirtual_aligned_free,
    &umemCudaManaged_set_,
    &umemCudaManaged_copy_to_,
    &umemCudaManaged_copy_from_,
  };
  assert(sizeof(CUdeviceptr) == sizeof(uintptr_t));
  umemHost_ctor(&ctx->host);
  umemVirtual_ctor(&ctx->super, &ctx->host);
  ctx->super.vptr = &vtbl;
  ctx->super.type = umemCudaManagedDevice;
  assert(cudaMemAttachHost!=0);
  ctx->flags = (flags ? flags : cudaMemAttachGlobal);
  ctx->async = async;
  ctx->stream = stream;

  // check for managed memory support:
  int count;
  CUDA_CALL(&ctx->super, cudaGetDeviceCount(&count),
	    umemRuntimeError, return,
	    "umemCudaManaged_ctor: cudaGetDeviceCount(&%d)",
	    count); 
  struct cudaDeviceProp prop;
  bool has_managedMemory = false;
  for (int device = 0; device < count; ++device) {
    CUDA_CALL(ctx, cudaGetDeviceProperties(&prop, device), umemRuntimeError, return,
              " umemCudaManaged_ctor:cudaGetDeviceProperties(%" PRIxPTR ", %d)", (uintptr_t)&prop, device);
    if (prop.managedMemory) {
      has_managedMemory = true;
      break;
    }
  }
  if (!has_managedMemory) {
      char buf[256];
      snprintf(buf, sizeof(buf),
               "umemCudaManaged_ctor: no devices found with managed memory support");
      umem_set_status(&ctx->super, umemNotSupportedError, buf);
  }
}
