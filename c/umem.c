#include <assert.h>
#include <string.h>
#include "umem.h"

/*
  umemVirtual virtual methods.
*/

static void umem_dtor_(umemVirtual  * const this) {
  assert(0); /* purely-virtual function should never be called */
}


static uintptr_t umem_alloc_(umemVirtual  * const this, size_t nbytes) {
  assert(0); /* purely-virtual function should never be called */
  return 0;
}

static bool umem_is_same_device_(umemVirtual  * const this, umemVirtual  * const she) {
  assert(0); /* purely-virtual function should never be called */
  return false;
}

static uintptr_t umem_calloc_(umemVirtual  * const this, size_t nmemb, size_t size) {
  assert(0); /* purely-virtual function should never be called */
  return 0;
}


static void umem_free_(umemVirtual  * const this, uintptr_t adr) {
  assert(0); /* purely-virtual function should never be called */
}


static uintptr_t umem_aligned_alloc_(umemVirtual  * const this, size_t alignement, size_t size) {
  assert(0); /* purely-virtual function should never be called */
  return 0;
}


static uintptr_t umem_aligned_origin_(umemVirtual  * const this, uintptr_t aligned_adr) {
  assert(0); /* purely-virtual function should never be called */
}


static void umem_aligned_free_(umemVirtual  * const this, uintptr_t aligned_adr) {
  assert(0); /* purely-virtual function should never be called */
}


static void umem_set_(umemVirtual * const this, uintptr_t adr, int c, size_t nbytes) {
  assert(0); /* purely-virtual function should never be called */
}


static void umem_copy_to_(umemVirtual * const this, uintptr_t src_adr,
			  umemVirtual * const that, uintptr_t dest_adr,
			  size_t nbytes) {
  assert(0); /* purely-virtual function should never be called */
}


static void umem_copy_from_(umemVirtual  * const this, uintptr_t src_adr,
			    umemVirtual  * const that, uintptr_t dest_adr,
			    size_t nbytes) {
  assert(0); /* purely-virtual function should never be called */
}


/*
  umemVirtual constructor.
*/
void umemVirtual_ctor(umemVirtual * const this, umemHost * host) {
  static struct umemVtbl const vtbl = {
    &umem_dtor_,
    &umem_is_same_device_,
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
  this->vptr = &vtbl;
  this->type = umemVirtualDevice;
  this->status.type = umemOK;
  // message is owned by umemVirtual instance. So, use only
  // umem_set_status or umem_clear_status to change it.
  this->status.message = NULL;
  this->host = (void*)host;
}

/*
  umemVirtual destructor.
*/
void umemVirtual_dtor(umemVirtual * const this) {
  if (this->status.message != NULL) {
    free(this->status.message);
    this->status.message = NULL;
  }
  this->status.type = umemOK;
  if (this->host != NULL) {
    umem_dtor(this->host);
    this->host = NULL;
  }
}

bool umemVirtual_is_same_device(umemVirtual * const this, umemVirtual * const that) {
  return false;
}


uintptr_t umemVirtual_calloc(umemVirtual * const this, size_t nmemb, size_t size) {
  uintptr_t adr = 0;
  if (size != 0) {
    size_t nbytes = nmemb * size; // TODO: check overflow
    adr = umem_alloc(this, nbytes);
    if (umem_is_ok(this))
      umem_set(this, adr, 0, nbytes);
  }
  return adr;
}



uintptr_t umemVirtual_aligned_alloc(umemVirtual * const this, size_t alignment, size_t size) {
  /*
    Requirements:
    1. alignment must be power of two
    2. size must be a multiple of alignment or zero
    3. alignement is at least fundamental alignment
   */
  uintptr_t adr = 0;
  size_t extra = (alignment - 1) + sizeof(uintptr_t);
  size_t req = extra + (size ? size: 1);
  adr = umem_calloc(this, req, 1);
  if (!umem_is_ok(this))
    return 0;
  uintptr_t aligned = adr + extra;
  aligned = aligned - (aligned % alignment);
  umem_copy_to(this->host, (uintptr_t)&adr, this, aligned-sizeof(uintptr_t), sizeof(uintptr_t));
  if (umem_is_ok(this))
    return aligned;
  umem_free(this, adr);
  return 0;
}

uintptr_t umemVirtual_aligned_origin(umemVirtual * const this, uintptr_t aligned_adr) {
  uintptr_t adr = 0;
  if (aligned_adr != 0) {
    umem_copy_from(this->host, (uintptr_t)&adr, this, aligned_adr-sizeof(uintptr_t), sizeof(uintptr_t));
    if (!umem_is_ok(this))
      return 0;
  }
  return adr;
}

void umemVirtual_aligned_free(umemVirtual * const this, uintptr_t aligned_adr) {
  umem_free(this, umemVirtual_aligned_origin(this, aligned_adr));
}

/*
  Status handling utility functions.
*/
void umem_set_status(void * const this,
		     umemStatusType type, const char * message) {
  umemVirtual * const this_ = this;
  if (message == NULL) {
    if (this_->status.message != NULL)
      free(this_->status.message);
    this_->status.message = NULL;
  } else {
    if (this_->status.message == NULL) {
      this_->status.message = strdup(message);
    } else {
      // append thisssage
      char buf[256];
      buf[0] = 0;
      if (this_->status.type != type) {
	snprintf(buf, sizeof(buf), "\nstatus %s changed to %s",
		 umem_get_status_name(this_->status.type),
		 umem_get_status_name(type));
      }
      size_t l1 = strlen(this_->status.message);
      size_t l2 = strlen(buf);
      size_t l3 = strlen(message);
      this_->status.message = realloc(this_->status.message,
				    l1 + l2 + l3 + 2);
      memcpy(this_->status.message + l1, buf, l2);
      memcpy(this_->status.message + l1 + l2, "\n", 1);
      memcpy(this_->status.message + l1 + l2 + 1, message, l3);
      this_->status.message[l1+l2+l3+1] = '\0';
    }
  }
  this_->status.type = type;
}


void umem_clear_status(void * const this) {
  umemVirtual * const this_ = this;
  if (this_->status.message != NULL) {
    free(this_->status.message);
    this_->status.message = NULL;
  }
  this_->status.type = umemOK;
}

/*
  Utility functions
*/

const char* umem_get_device_name(umemDeviceType type) {
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
  switch (type) {
  case umemOK: return ok_name;
  case umemMemoryError: return memory_error_name;
  case umemRuntimeError: return runtime_error_name;
  case umemIOError: return io_error_name;
  case umemNotImplementedError: return notimpl_error_name;
  case umemAssertError: return assert_error_name;
  case umemValueError: return value_error_name;
  case umemTypeError: return type_error_name;
  }
  return NULL;
}
