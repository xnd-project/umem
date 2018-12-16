#include <assert.h>
#include <string.h>
#include "umem.h"

/*
  umemVirtual virtual methods.
*/

static void umem_dtor_(umemVirtual  * const ctx) {
  assert(0); /* purely-virtual function should never be called */
}


static uintptr_t umem_alloc_(umemVirtual  * const ctx, size_t nbytes) {
  assert(0); /* purely-virtual function should never be called */
  return 0;
}

static bool umem_is_same_context_(umemVirtual  * const ctx, umemVirtual  * const she) {
  assert(0); /* purely-virtual function should never be called */
  return false;
}

static uintptr_t umem_calloc_(umemVirtual  * const ctx, size_t nmemb, size_t size) {
  assert(0); /* purely-virtual function should never be called */
  return 0;
}


static void umem_free_(umemVirtual  * const ctx, uintptr_t adr) {
  assert(0); /* purely-virtual function should never be called */
}


static uintptr_t umem_aligned_alloc_(umemVirtual  * const ctx, size_t alignement, size_t size) {
  assert(0); /* purely-virtual function should never be called */
  return 0;
}


static uintptr_t umem_aligned_origin_(umemVirtual  * const ctx, uintptr_t aligned_adr) {
  assert(0); /* purely-virtual function should never be called */
  return 0;
}


static void umem_aligned_free_(umemVirtual  * const ctx, uintptr_t aligned_adr) {
  assert(0); /* purely-virtual function should never be called */
}


static void umem_set_(umemVirtual * const ctx, uintptr_t adr, int c, size_t nbytes) {
  assert(0); /* purely-virtual function should never be called */
}


static void umem_copy_to_(umemVirtual * const src_ctx, uintptr_t src_adr,
			  umemVirtual * const dest_ctx, uintptr_t dest_adr,
			  size_t nbytes) {
  assert(0); /* purely-virtual function should never be called */
}


static void umem_copy_from_(umemVirtual  * const dest_ctx, uintptr_t dest_adr,
			    umemVirtual  * const src_ctx, uintptr_t src_adr,
			    size_t nbytes) {
  assert(0); /* purely-virtual function should never be called */
}


/*
  umemVirtual constructor.
*/
void umemVirtual_ctor(umemVirtual * const ctx, umemHost * host_ctx) {
  static struct umemVtbl const vtbl = {
    &umem_dtor_,
    &umem_is_same_context_,
    &umem_alloc_,
    &umem_calloc_,
    &umem_free_,
    &umem_aligned_alloc_,
    &umem_aligned_origin_,
    &umem_aligned_free_,
    &umem_set_,
    &umem_copy_to_,
    &umem_copy_from_,
  };
  ctx->vptr = &vtbl;
  ctx->type = umemVirtualDevice;
  ctx->status.type = umemOK;
  // message is owned by umemVirtual instance. So, use only
  // umem_set_status or umem_clear_status to change it.
  ctx->status.message = NULL;
  ctx->host_ctx = (void*)host_ctx;
}

/*
  umemVirtual destructor.
*/
void umemVirtual_dtor(umemVirtual * const ctx) {
  if (ctx->status.message != NULL) {
    free(ctx->status.message);
    ctx->status.message = NULL;
  }
  ctx->status.type = umemOK;
  if (ctx->host_ctx != NULL) {
    umem_dtor(ctx->host_ctx);
    ctx->host_ctx = NULL;
  }
}

bool umemVirtual_is_same_context(umemVirtual * const one_ctx, umemVirtual * const other_ctx) {
  return one_ctx == other_ctx;
}

uintptr_t umemVirtual_calloc(umemVirtual * const ctx, size_t nmemb, size_t size) {
  uintptr_t adr = 0;
  if (size != 0) {
    size_t nbytes = nmemb * size; // TODO: check overflow
    adr = umem_alloc(ctx, nbytes);
    if (umem_is_ok(ctx))
      umem_set(ctx, adr, 0, nbytes);
  }
  return adr;
}

uintptr_t umemVirtual_aligned_alloc(umemVirtual * const ctx, size_t alignment, size_t size) {
  /*
    Requirements:
    1. alignment must be power of two
    2. size must be a multiple of alignment or zero
    3. alignement is at least fundamental alignment
   */
  uintptr_t adr = 0;
  size_t extra = (alignment - 1) + sizeof(uintptr_t);
  size_t req = extra + (size ? size: 1);
  adr = umem_calloc(ctx, req, 1);
  if (!umem_is_ok(ctx))
    return 0;
  uintptr_t aligned = adr + extra;
  aligned = aligned - (aligned % alignment);
  umem_copy_to(ctx->host_ctx, (uintptr_t)&adr, ctx, aligned-sizeof(uintptr_t), sizeof(uintptr_t));
  if (umem_is_ok(ctx))
    return aligned;
  umem_free(ctx, adr);
  return 0;
}

uintptr_t umemVirtual_aligned_origin(umemVirtual * const ctx, uintptr_t aligned_adr) {
  uintptr_t adr = 0;
  if (aligned_adr != 0) {
    umem_copy_from(ctx->host_ctx, (uintptr_t)&adr, ctx, aligned_adr-sizeof(uintptr_t), sizeof(uintptr_t));
    if (!umem_is_ok(ctx))
      return 0;
  }
  return adr;
}

void umemVirtual_aligned_free(umemVirtual * const ctx, uintptr_t aligned_adr) {
  umem_free(ctx, umemVirtual_aligned_origin(ctx, aligned_adr));
}

/*
  Status handling utility functions.
*/
void umem_set_status(void * const ctx,
		     umemStatusType type, const char * message) {
  umemVirtual * const ctx_ = ctx;
  if (message == NULL) {
    if (ctx_->status.message != NULL)
      free(ctx_->status.message);
    ctx_->status.message = NULL;
  } else {
    if (ctx_->status.message == NULL) {
      ctx_->status.message = strdup(message);
    } else {
      // append ctxssage
      char buf[256];
      buf[0] = 0;
      if (ctx_->status.type != type) {
	snprintf(buf, sizeof(buf), "\nstatus %s changed to %s",
		 umem_get_status_name(ctx_->status.type),
		 umem_get_status_name(type));
      }
      size_t l1 = strlen(ctx_->status.message);
      size_t l2 = strlen(buf);
      size_t l3 = strlen(message);
      ctx_->status.message = realloc(ctx_->status.message,
				    l1 + l2 + l3 + 2);
      memcpy(ctx_->status.message + l1, buf, l2);
      memcpy(ctx_->status.message + l1 + l2, "\n", 1);
      memcpy(ctx_->status.message + l1 + l2 + 1, message, l3);
      ctx_->status.message[l1+l2+l3+1] = '\0';
    }
  }
  ctx_->status.type = type;
}


void umem_clear_status(void * const ctx) {
  umemVirtual * const ctx_ = ctx;
  if (ctx_->status.message != NULL) {
    free(ctx_->status.message);
    ctx_->status.message = NULL;
  }
  ctx_->status.type = umemOK;
}

/*
  Utility functions
*/

const char* umem_get_device_name_from_type(umemDeviceType type) {
  static const char virtual_device_name[] = "Virtual";
  static const char host_device_name[] = "Host";
  static const char file_device_name[] = "FILE";
  static const char cuda_device_name[] = "CUDA";
  static const char mmap_device_name[] = "MMAP"; // NOT IMPLEMENTED
  static const char rmm_device_name[] = "RRM"; // NOT IMPLEMENTED
  switch(type) {
  case umemVirtualDevice: return virtual_device_name;
  case umemHostDevice: return host_device_name;
  case umemFileDevice: return file_device_name;
  case umemCudaDevice: return cuda_device_name;
  case umemMMapDevice: return mmap_device_name;
  case umemRMMDevice: return rmm_device_name;
  }
  return NULL;
}


const char* umem_get_status_name(umemStatusType type) {
  static const char ok_name[] = "OK";
  static const char memory_error_name[] = "MemoryError";
  static const char runtime_error_name[] = "RuntimeError";
  static const char io_error_name[] = "IOError";
  static const char notimpl_error_name[] = "NotImplementedError";
  static const char assert_error_name[] = "AssertError";
  static const char value_error_name[] = "ValueError";
  static const char type_error_name[] = "TypeError";
  static const char index_error_name[] = "IndexError";
  switch (type) {
  case umemOK: return ok_name;
  case umemMemoryError: return memory_error_name;
  case umemRuntimeError: return runtime_error_name;
  case umemIOError: return io_error_name;
  case umemNotImplementedError: return notimpl_error_name;
  case umemAssertError: return assert_error_name;
  case umemValueError: return value_error_name;
  case umemTypeError: return type_error_name;
  case umemIndexError: return index_error_name;
  }
  return NULL;
}
