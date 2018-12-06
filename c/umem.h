#ifndef UMEM_H
#define UMEM_H
/*
  Author: Pearu Peterson
  Created: November 2018
*/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "umem_utils.h"

/*
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
  umemUMMDevice,  // not impl
} umemDeviceType;


/*
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
  umemNotImplementedError,
  umemAssertError,
} umemStatusType;


typedef struct {
  umemStatusType type;
  char* message;
} umemStatus;


/*
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

/*
  umemHost represents host memory.

  Public API.
*/
typedef struct {
  umemVirtual super;
} umemHost;


UMEM_EXTERN void umemVirtual_ctor(umemVirtual * const me, umemHost* host); /* Constructor. Internal API */


UMEM_EXTERN void umemVirtual_dtor(umemVirtual * const me); /* Destructor. Internal API */


/*
  Status and error handling functions.

  Public API.
*/

static inline umemStatusType umem_get_status(void * const me) {
  return ((umemVirtual * const)me)->status.type;
}


static inline const char * umem_get_message(void * const me) {
  return (((umemVirtual * const)me)->status.message == NULL ?
	  "" : ((umemVirtual * const)me)->status.message); }


static inline int umem_is_ok(void * const me) {
  return ((umemVirtual * const)me)->status.type == umemOK;
}


UMEM_EXTERN void umem_set_status(void * const me,
			    umemStatusType type, const char * message);


UMEM_EXTERN void umem_clear_status(void * const me);





UMEM_EXTERN void umemHost_ctor(umemHost * const me);  /* Constructor. Public API. */

/*
  umemVtbl defines a table of umemVirtual methods.

  Internal API.
*/
struct umemVtbl {
  void (*dtor)(umemVirtual * const me);
  bool (*is_same_device)(umemVirtual * const me, umemVirtual * const she);
  uintptr_t (*alloc)(umemVirtual * const me, size_t nbytes);
  uintptr_t (*calloc)(umemVirtual * const me, size_t nmemb, size_t size);
  void (*free)(umemVirtual * const me, uintptr_t adr);
  uintptr_t (*aligned_alloc)(umemVirtual * const me, size_t alignment, size_t size);
  uintptr_t (*aligned_origin)(umemVirtual * const me, uintptr_t aligned_adr);
  void (*aligned_free)(umemVirtual * const me, uintptr_t aligned_adr);
  void (*set)(umemVirtual * const me, uintptr_t adr, int c, size_t nbytes);
  void (*copy_to)(umemVirtual * const me, uintptr_t src_adr,
		  umemVirtual * const she, uintptr_t dest_adr,
		  size_t nbytes);
  void (*copy_from)(umemVirtual * const me, uintptr_t dest_adr,
		    umemVirtual * const she, uintptr_t src_adr,
		    size_t nbytes);
};


#ifdef HAVE_CUDA
/*
  umemCuda represents GPU device memory using CUDA library.

  Public API.
*/
typedef struct {
  umemVirtual super;
  int device;
} umemCuda;


UMEM_EXTERN void umemCuda_ctor(umemCuda * const me, int device); /* Constructor. Public API. */

#endif


/*
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


UMEM_EXTERN void umemFile_ctor(umemFile * const me,
			  const char* filename, const char* mode);  /* Constructor. Public API. */


/*
  umem_dtor is device context destructor

  Public API.
*/
static inline void umem_dtor(void * const me) {
  (*((umemVirtual * const)me)->vptr->dtor)(me);
}

/*
  umem_is_same_device returns true if the devices are the same in the
  sense of memory address spaces.

  Public/Internal API.
 */
static inline bool umem_is_same_device(void * const me, void * const she) {
  return (me == she ? true :
          ((((umemVirtual * const)me)->type == ((umemVirtual * const)she)->type
            ? (*((umemVirtual * const)me)->vptr->is_same_device)(me, she)
            : false)));
}

/*
  umem_alloc allocates device memory and returns the memory addresss.

  Public API.
*/
static inline uintptr_t umem_alloc(void * const me, size_t nbytes) {
  return (*((umemVirtual * const)me)->vptr->alloc)(me, nbytes);
}

static inline uintptr_t umem_calloc(void * const me, size_t nmemb, size_t size) {
  return (*((umemVirtual * const)me)->vptr->calloc)(me, nmemb, size);
}

static inline size_t umem_fundamental_align(void * const me) {
  switch (((umemVirtual * const)me)->type) {
  case umemFileDevice: return UMEM_FUNDAMENTAL_FILE_ALIGN;
  case umemCudaDevice: return UMEM_FUNDAMENTAL_CUDA_ALIGN;
  case umemHostDevice: return UMEM_FUNDAMENTAL_HOST_ALIGN;
  }
  return 1;
}

static inline uintptr_t umem_aligned_alloc(void * const me, size_t alignment, size_t size) {
  TRY_RETURN(me, !umem_ispowerof2(alignment), umemValueError, return 0,
             "umemVirtual_aligned_alloc: alignment %zu must be power of 2",
             alignment);
  TRY_RETURN(me, size % alignment, umemValueError, return 0,
             "umemVirtual_aligned_alloc: size %zu must be multiple of alignment %zu",
             size, alignment);
  size_t fundamental_align = umem_fundamental_align(me);
  alignment = (alignment < fundamental_align ? fundamental_align : alignment);
  return (*((umemVirtual * const)me)->vptr->aligned_alloc)(me, alignment, size);
}

/*
  umem_aligned_origin returns starting address of memory containing
  the aligned memory area. This starting address can be used to free
  the aligned memory.
 */
static inline uintptr_t umem_aligned_origin(void * const me, uintptr_t aligned_adr) {
  return (*((umemVirtual * const)me)->vptr->aligned_origin)(me, aligned_adr);
}

/*
  umem_free frees device memory that was allocated using umem_alloc or umem_calloc.

  Public API.
*/
static inline void umem_free(void * const me, uintptr_t adr) {
  (*((umemVirtual * const)me)->vptr->free)(me, adr);
}


/*
  umem_aligned_free frees device memory that was allocated using umem_aligned_alloc.

  Public API.
*/
static inline void umem_aligned_free(void * const me, uintptr_t aligned_adr) {
  (*((umemVirtual * const)me)->vptr->aligned_free)(me, aligned_adr);
}


/*
  umem_set fills memory with constant byte

  Public API.
*/
static inline void umem_set(void * const me, uintptr_t adr, int c, size_t nbytes) {
  (*((umemVirtual * const)me)->vptr->set)(me, adr, c, nbytes);
}


/*
  umem_copy_from and umem_copy_to copy data from one device to another.

  Public API.
*/
static inline void umem_copy_to(void * const me, uintptr_t src_adr,
				void * const she, uintptr_t dest_adr,
				size_t nbytes) {
  (*((umemVirtual * const)me)->vptr->copy_to)(me, src_adr,
					      she, dest_adr, nbytes);
}


static inline void umem_copy_from(void * const me, uintptr_t dest_adr,
				  void * const she, uintptr_t src_adr,
				  size_t nbytes) {
  (*((umemVirtual * const)me)->vptr->copy_from)(me, dest_adr,
						she, src_adr, nbytes);
}


/*
  Methods for copying data from one device to another using host
  memory as a buffer.

  Internal/Public API.
*/
UMEM_EXTERN void umem_copy_to_via_host(void * const me, uintptr_t src_adr,
                                       void * const she, uintptr_t dest_adr,
                                       size_t nbytes);

UMEM_EXTERN void umem_copy_from_via_host(void * const me, uintptr_t dest_adr,
                                         void * const she, uintptr_t src_adr,
                                         size_t nbytes);

/*
  Methods for syncing data between devices.

  Public API.
 */

UMEM_EXTERN uintptr_t umem_connect(void * const src, uintptr_t src_adr,
                                   size_t nbytes, void * const dest);
UMEM_EXTERN void umem_sync_from(void * const src, uintptr_t src_adr,
                                void * const dest, uintptr_t dest_adr,
                                size_t nbytes);
UMEM_EXTERN void umem_sync_to(void * const src, uintptr_t src_adr,
                              void * const dest, uintptr_t dest_adr,
                              size_t nbytes);
UMEM_EXTERN void umem_disconnect(void * const src, void * const dest,
                                 uintptr_t dest_adr);


/*
  Various utility functions.

  Public API.
*/
UMEM_EXTERN const char* umem_get_device_name(umemDeviceType type);


UMEM_EXTERN const char* umem_get_status_name(umemStatusType type);


/*
  Generic methods and utility functions.

  Internal API.
*/

UMEM_EXTERN bool umemVirtual_is_same_device(umemVirtual * const me, umemVirtual * const she);

UMEM_EXTERN uintptr_t umemVirtual_calloc(umemVirtual * const me,
                                         size_t nmemb, size_t size);

UMEM_EXTERN uintptr_t umemVirtual_aligned_alloc(umemVirtual * const me,
                                                size_t alignment,
                                                size_t size);

UMEM_EXTERN void umemVirtual_aligned_free(umemVirtual * const me,
                                          uintptr_t aligned_adr);

UMEM_EXTERN uintptr_t umemVirtual_aligned_origin(umemVirtual * const me, uintptr_t aligned_adr);


#endif
