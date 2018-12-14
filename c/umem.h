/*! \mainpage
 *
 * See [UMEM homepage](https://github.com/plures/umem/) for more
 * information about this project.
 *
 */

#ifndef UMEM_H
#define UMEM_H
/*
  Author: Pearu Peterson
  Created: November 2018
*/

#ifdef __cplusplus
#define START_EXTERN_C extern "C" {
#define CLOSE_EXTERN_C }
#define this this_
#else
#define START_EXTERN_C
#define CLOSE_EXTERN_C
#endif

START_EXTERN_C

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "umem_config.h"
#include "umem_utils.h"

/**
   umemDeviceType defines the flags of supported device memory
   classes.
   
   Public API.
*/
typedef enum {
  umemVirtualDevice = 0,
  umemHostDevice,
  umemFileDevice,
  umemCudaDevice,
  umemMMapDevice, // not impl
  umemRMMDevice,  // not impl
} umemDeviceType;


/**
   umemStatusType defines the type flags for umemStatus.
   
   Public API.
*/

typedef enum {
  umemOK = 0,
  umemMemoryError,
  umemRuntimeError,
  umemIOError,
  umemValueError,
  umemTypeError,
  umemIndexError,
  umemNotImplementedError,
  umemAssertError,
} umemStatusType;


typedef struct {
  umemStatusType type;
  char* message;
} umemStatus;


/**
  umem is the "base" class for all device memory classes.

  Internal API.
*/
struct umemVtbl;
typedef struct {
  struct umemVtbl const *vptr;
  umemDeviceType type;
  umemStatus status;
  void* host;
} umemVirtual;

/**
  umemHost represents host memory.

  Public API.
*/
typedef struct {
  umemVirtual super;
} umemHost;


UMEM_EXTERN void umemVirtual_ctor(umemVirtual * const this, umemHost* host); /* Constructor. Internal API */


UMEM_EXTERN void umemVirtual_dtor(umemVirtual * const this); /* Destructor. Internal API */


/**
  Status and error handling functions.

  Public API.
*/

static inline umemStatusType umem_get_status(void * const this) {
  return ((umemVirtual * const)this)->status.type;
}


static inline const char * umem_get_message(void * const this) {
  return (((umemVirtual * const)this)->status.message == NULL ?
	  "" : ((umemVirtual * const)this)->status.message); }


static inline bool umem_is_ok(void * const this) {
  return ((umemVirtual * const)this)->status.type == umemOK;
}


UMEM_EXTERN void umem_set_status(void * const this,
                                 umemStatusType type, const char * message);


UMEM_EXTERN void umem_clear_status(void * const this);





UMEM_EXTERN void umemHost_ctor(umemHost * const this);  /* Constructor. Public API. */

/**
  umemVtbl defines a table of umemVirtual methods.

  Internal API.
*/
struct umemVtbl {
  void (*dtor)(umemVirtual * const this);
  bool (*is_same_device)(umemVirtual * const this, umemVirtual * const that);
  uintptr_t (*alloc)(umemVirtual * const this, size_t nbytes);
  uintptr_t (*calloc)(umemVirtual * const this, size_t nmemb, size_t size);
  void (*free)(umemVirtual * const this, uintptr_t adr);
  uintptr_t (*aligned_alloc)(umemVirtual * const this, size_t alignment, size_t size);
  uintptr_t (*aligned_origin)(umemVirtual * const this, uintptr_t aligned_adr);
  void (*aligned_free)(umemVirtual * const this, uintptr_t aligned_adr);
  void (*set)(umemVirtual * const this, uintptr_t adr, int c, size_t nbytes);
  void (*copy_to)(umemVirtual * const this, uintptr_t src_adr,
		  umemVirtual * const that, uintptr_t dest_adr,
		  size_t nbytes);
  void (*copy_from)(umemVirtual * const this, uintptr_t dest_adr,
		    umemVirtual * const that, uintptr_t src_adr,
		    size_t nbytes);
};


#ifdef HAVE_CUDA
/**
  umemCuda represents GPU device memory using CUDA library.

  Public API.
*/
typedef struct {
  umemVirtual super;
  umemHost host;
  int device;
} umemCuda;


UMEM_EXTERN void umemCuda_ctor(umemCuda * const this, int device); /* Constructor. Public API. */

#endif


/**
  umemFile represents FILE in binary format.

  alloc - opens the file with given mode, nbytes argument is ignored
  free  - closes the file
  adr   - is the file position relative to the start of the file

  Public API.
*/
typedef struct {
  umemVirtual super;
  umemHost host;
  uintptr_t fp;
  const char * filename;
  const char * mode;
} umemFile;


UMEM_EXTERN void umemFile_ctor(umemFile * const this,
			  const char* filename, const char* mode);  /* Constructor. Public API. */


/**
  umem_dtor is device context destructor

  Public API.
*/
static inline void umem_dtor(void * const this) {
  (*((umemVirtual * const)this)->vptr->dtor)((umemVirtual * const)this);
}

/**
  umem_is_same_device returns true if the devices are the same in the
  sense of memory address spaces.

  Public/Internal API.
 */
static inline bool umem_is_same_device(void * const this, void * const that) {
  return (this == that ? true :
          ((((umemVirtual * const)this)->type == ((umemVirtual * const)that)->type
            ? (*((umemVirtual * const)this)->vptr->is_same_device)((umemVirtual * const)this, (umemVirtual * const)that)
            : false)));
}

/**
  umem_alloc allocates device memory and returns the memory addresss.

  Public API.
*/
static inline uintptr_t umem_alloc(void * const this, size_t nbytes) {
  return (*((umemVirtual * const)this)->vptr->alloc)((umemVirtual * const)this, nbytes);
}

static inline uintptr_t umem_calloc(void * const this, size_t nmemb, size_t size) {
  return (*((umemVirtual * const)this)->vptr->calloc)((umemVirtual * const)this, nmemb, size);
}

static inline size_t umem_fundamental_align(void * const this) {
  switch (((umemVirtual * const)this)->type) {
  case umemFileDevice: return UMEM_FUNDAMENTAL_FILE_ALIGN;
  case umemCudaDevice: return UMEM_FUNDAMENTAL_CUDA_ALIGN;
  case umemHostDevice: return UMEM_FUNDAMENTAL_HOST_ALIGN;
  default: ;
  }
  return 1;
}

static inline uintptr_t umem_aligned_alloc(void * const this,
                                           size_t alignment, size_t size)
{
  TRY_RETURN(this, !umem_ispowerof2(alignment), umemValueError, return 0,
             "umem_aligned_alloc: alignment %zu must be power of 2",
             alignment);
  TRY_RETURN(this, size % alignment, umemValueError, return 0,
             "umem_aligned_alloc: size %zu must be multiple of alignment %zu",
             size, alignment);
  size_t fundamental_align = umem_fundamental_align(this);
  alignment = (alignment < fundamental_align ? fundamental_align : alignment);
  return (*((umemVirtual * const)this)->vptr->aligned_alloc)
    ((umemVirtual * const)this, alignment, size);
}

/**
  umem_aligned_origin returns starting address of memory containing
  the aligned memory area. This starting address can be used to free
  the aligned memory.
 */
static inline uintptr_t umem_aligned_origin(void * const this,
                                            uintptr_t aligned_adr) {
  return (*((umemVirtual * const)this)->vptr->aligned_origin)
    ((umemVirtual * const)this, aligned_adr);
}

/**
  umem_free frees device memory that was allocated using umem_alloc or umem_calloc.

  Public API.
*/
static inline void umem_free(void * const this, uintptr_t adr) {
  (*((umemVirtual * const)this)->vptr->free)((umemVirtual * const)this, adr);
}


/**
  umem_aligned_free frees device memory that was allocated using umem_aligned_alloc.

  Public API.
*/
static inline void umem_aligned_free(void * const this, uintptr_t aligned_adr) {
  (*((umemVirtual * const)this)->vptr->aligned_free)((umemVirtual * const)this, aligned_adr);
}


/**
  umem_set fills memory with constant byte

  Public API.
*/
static inline void umem_set(void * const this, uintptr_t adr, int c, size_t nbytes) {
  (*((umemVirtual * const)this)->vptr->set)((umemVirtual * const)this, adr, c, nbytes);
}

static inline void umem_set_safe(void * const this, uintptr_t adr, size_t size, int c, size_t nbytes) {
  if (size < nbytes)
    umem_set_status(this, umemIndexError, "umem_set_safe: nbytes out of range");
  else
    umem_set(this, adr, c, nbytes);
}

/**
  umem_copy_from and umem_copy_to copy data from one device to another.

  Public API.
*/
static inline void umem_copy_to(void * const this, uintptr_t src_adr,
				void * const that, uintptr_t dest_adr,
				size_t nbytes) {
  (*((umemVirtual * const)this)->vptr->copy_to)((umemVirtual * const)this, src_adr,
                                                (umemVirtual * const)that, dest_adr, nbytes);
}

static inline void umem_copy_to_safe(void * const this, uintptr_t src_adr, size_t src_size,
                                     void * const that, uintptr_t dest_adr, size_t dest_size,
                                     size_t nbytes) {
  if (!(nbytes<=src_size && nbytes<=dest_size))
    umem_set_status(this, umemIndexError, "umem_copy_to_safe: nbytes out of range");
  else
    umem_copy_to(this, src_adr, that, dest_adr, nbytes);
}

static inline void umem_copy_from(void * const this, uintptr_t dest_adr,
				  void * const that, uintptr_t src_adr,
				  size_t nbytes) {
  (*((umemVirtual * const)this)->vptr->copy_from)((umemVirtual * const)this, dest_adr,
                                                  (umemVirtual * const)that, src_adr, nbytes);
}

static inline void umem_copy_from_safe(void * const this, uintptr_t dest_adr, size_t dest_size,
                                       void * const that, uintptr_t src_adr, size_t src_size,
                                       size_t nbytes) {
  if (!(nbytes<=src_size && nbytes<=dest_size))
    umem_set_status(this, umemIndexError, "umem_copy_from_safe: nbytes out of range");
  else
    umem_copy_from(this, dest_adr, that, src_adr, nbytes);
}


/**
  Methods for copying data from one device to another using host
  memory as a buffer.

  Internal/Public API.
*/
UMEM_EXTERN void umem_copy_to_via_host(void * const this, uintptr_t src_adr,
                                       void * const that, uintptr_t dest_adr,
                                       size_t nbytes);

UMEM_EXTERN void umem_copy_from_via_host(void * const this, uintptr_t dest_adr,
                                         void * const that, uintptr_t src_adr,
                                         size_t nbytes);

/**
  Methods for syncing data between devices.

  Public API.
 */

UMEM_EXTERN uintptr_t umem_connect(void * const src, uintptr_t src_adr,
                                   size_t nbytes, void * const dest,
                                   size_t dest_alignment);
UMEM_EXTERN void umem_sync_from(void * const src, uintptr_t src_adr,
                                void * const dest, uintptr_t dest_adr,
                                size_t nbytes);
UMEM_EXTERN void umem_sync_to(void * const src, uintptr_t src_adr,
                              void * const dest, uintptr_t dest_adr,
                              size_t nbytes);
UMEM_EXTERN void umem_disconnect(void * const src, uintptr_t src_adr,
                                 void * const dest, uintptr_t dest_adr,
                                 size_t dest_alignment);

static inline void umem_sync_from_safe(void * const dest, uintptr_t dest_adr, size_t dest_size,
                                       void * const src, uintptr_t src_adr, size_t src_size,
                                       size_t nbytes) {
  if (!(nbytes<=src_size && nbytes<=dest_size))
    umem_set_status(dest, umemIndexError, "umem_sync_from_safe: nbytes out of range");
  else
    umem_sync_from(dest, dest_adr, src, src_adr, nbytes);
}

static inline void umem_sync_to_safe(void * const src, uintptr_t src_adr, size_t src_size,
                                     void * const dest, uintptr_t dest_adr, size_t dest_size,
                                     size_t nbytes) {
  if (!(nbytes<=src_size && nbytes<=dest_size))
    umem_set_status(src, umemIndexError, "umem_sync_to_safe: nbytes out of range");
  else
    umem_sync_to(src, src_adr, dest, dest_adr, nbytes);
}

  
/**
  Various utility functions.

  Public API.
*/
UMEM_EXTERN const char* umem_get_device_name(umemDeviceType type);


UMEM_EXTERN const char* umem_get_status_name(umemStatusType type);


/**
  Generic methods and utility functions.

  Internal API.
*/

UMEM_EXTERN bool umemVirtual_is_same_device(umemVirtual * const this, umemVirtual * const that);

UMEM_EXTERN uintptr_t umemVirtual_calloc(umemVirtual * const this,
                                         size_t nmemb, size_t size);

UMEM_EXTERN uintptr_t umemVirtual_aligned_alloc(umemVirtual * const this,
                                                size_t alignment,
                                                size_t size);

UMEM_EXTERN void umemVirtual_aligned_free(umemVirtual * const this,
                                          uintptr_t aligned_adr);

UMEM_EXTERN uintptr_t umemVirtual_aligned_origin(umemVirtual * const this, uintptr_t aligned_adr);

CLOSE_EXTERN_C

#ifdef __cplusplus
#undef this
#include "umem.hpp"
#endif

#endif
