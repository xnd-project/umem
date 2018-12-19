/** 
    @mainpage 

    See [UMEM homepage](https://github.com/plures/umem/) for more
    information about the UMEM project.

    All C API functions are documented in [libumem C API
    documentation](https://umem.readthedocs.io/)

    @author Pearu Peterson
    @date November 2018
*/

#ifndef UMEM_H
#define UMEM_H

#ifdef __cplusplus
#define START_EXTERN_C extern "C" {
#define CLOSE_EXTERN_C }
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
*/
typedef enum {
  umemVirtualDevice = 0,
  umemHostDevice,
  umemFileDevice,
  umemCudaDevice,
  umemMMapDevice,
  umemRMMDevice,
} umemDeviceType;


/**
   umemStatusType defines the type flags for umemStatus.
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

/**
   umemStatus holds the status type and message of a memory context
   type.
*/
typedef struct {
  umemStatusType type;
  char* message;
} umemStatus;

/**
   umemVirtual is the "base" type for all memory context types.
*/
struct umemVtbl;
typedef struct {
  struct umemVtbl const *vptr;
  umemDeviceType type;
  umemStatus status;
  void* host_ctx;
} umemVirtual;

/**
   umemVtbl defines a table of umemVirtual methods.
*/
struct umemVtbl {
  /** @brief Holds pointer to object destructor. */
  void (*dtor)(umemVirtual * const ctx);
  /** @brief Holds pointer to object is_same_context method. */
  bool (*is_same_context)(umemVirtual * const one_ctx, umemVirtual * const other_ctx);
  /** @brief Holds pointer to object alloc method. */
  uintptr_t (*alloc)(umemVirtual * const ctx, size_t nbytes);
  /** @brief Holds pointer to object calloc method. */
  uintptr_t (*calloc)(umemVirtual * const ctx, size_t nmemb, size_t size);
  /** @brief Holds pointer to object free method. */
  void (*free)(umemVirtual * const ctx, uintptr_t adr);
  /** @brief Holds pointer to object aligned_alloc method. */
  uintptr_t (*aligned_alloc)(umemVirtual * const ctx, size_t alignment, size_t size);
  /** @brief Holds pointer to object aligned_origin method. */
  uintptr_t (*aligned_origin)(umemVirtual * const ctx, uintptr_t aligned_adr);
  /** @brief Holds pointer to object aligned_free method. */
  void (*aligned_free)(umemVirtual * const ctx, uintptr_t aligned_adr);
  /** @brief Holds pointer to object set method. */
  void (*set)(umemVirtual * const ctx, uintptr_t adr, int c, size_t nbytes);
  /** @brief Holds pointer to object copy_to method. */
  void (*copy_to)(umemVirtual * const src_ctx, uintptr_t src_adr,
		  umemVirtual * const dest_ctx, uintptr_t dest_adr,
		  size_t nbytes);
  /** @brief Holds pointer to object copy_from method. */
  void (*copy_from)(umemVirtual * const dest_ctx, uintptr_t dest_adr,
		    umemVirtual * const src_ctx, uintptr_t src_adr,
		    size_t nbytes);
};



/**
   umemHost represents host memory context. 

   To construct/destruct the umemHost object, use
   umemHost_ctor/umem_dtor.
*/
typedef struct {
  umemVirtual super;
} umemHost;

UMEM_EXTERN void umemHost_ctor(umemHost * const ctx);

UMEM_EXTERN void umemVirtual_ctor(umemVirtual * const ctx, umemHost* host_ctx);

#ifdef HAVE_CUDA_CONTEXT
/**
   umemCuda represents GPU device memory using CUDA library. 

   To construct/destruct the umemCuda object, use
   umemCuda_ctor/umem_dtor.
*/
typedef struct {
  umemVirtual super;
  umemHost host;
  int device;
} umemCuda;

UMEM_EXTERN void umemCuda_ctor(umemCuda * const ctx, int device);

#endif

#ifdef HAVE_RMM_CONTEXT
/**
   umemRMM represents GPU device memory using rmm from rapidsay/cudf. 

   To construct/destruct the umemRMM object, use
   umemRMM_ctor/umem_dtor.
*/
typedef struct {
  umemVirtual super;
  umemHost host;
  uintptr_t stream;
} umemRMM;

UMEM_EXTERN void umemRMM_ctor(umemRMM * const ctx, uintptr_t stream);

#endif

#ifdef HAVE_FILE_CONTEXT
/**
  umemFile represents a stdio.h based file in binary format.

   To construct/destruct the umemFile object, use
   umemFile_ctor/umem_dtor.
*/
typedef struct {
  umemVirtual super;
  umemHost host;
  uintptr_t fp;
  const char * filename;
  const char * mode;
} umemFile;

UMEM_EXTERN void umemFile_ctor(umemFile * const ctx,
                               const char* filename, const char* mode);

#endif

#ifdef HAVE_MMAP_CONTEXT
/**
  umemMMap represents a sys/mman.h based memory mapped file.

   To construct/destruct the umemMMap object, use
   umemMMap_ctor/umem_dtor.
*/
typedef struct {
  umemVirtual super;
  umemHost host;
  uintptr_t fp;
  const char * filename;
  const char * mode;
} umemMMap;

UMEM_EXTERN void umemMMap_ctor(umemMMap * const ctx,
                               const char* filename, const char* mode);

#endif

/**
   Public C API functions.
*/

/**
   Memory context destructor.
*/
static inline void umem_dtor(void * const ctx);

/**
   umem_alloc allocates device memory and returns the memory addresss.
*/
static inline uintptr_t umem_alloc(void * const ctx, size_t nbytes);
static inline uintptr_t umem_calloc(void * const ctx, size_t nmemb, size_t size);
static inline
uintptr_t umem_aligned_alloc(void * const ctx, size_t alignment, size_t size);
/**
  umem_aligned_origin returns starting address of memory containing
  the aligned memory area. This starting address can be used to free
  the aligned memory.
*/
static inline
uintptr_t umem_aligned_origin(void * const ctx,
                              uintptr_t aligned_adr);
static inline 
bool umem_is_same_context(void * const one_ctx, void * const other_ctx);
static inline
size_t umem_fundamental_align(void * const ctx);

/**
   umem_free frees device memory that was allocated using umem_alloc
   or umem_calloc.
*/
static inline
void umem_free(void * const ctx, uintptr_t adr);
static inline
void umem_aligned_free(void * const ctx, uintptr_t aligned_adr);
/**
  umem_set fills memory with constant byte
*/
static inline
void umem_set(void * const ctx, uintptr_t adr, int c, size_t nbytes);
static inline
void umem_set_safe(void * const ctx, uintptr_t adr, size_t size, int c, size_t nbytes);
static inline 
void umem_copy_to(void * const src_ctx, uintptr_t src_adr,
                  void * const dest_ctx, uintptr_t dest_adr,
                  size_t nbytes);
static inline 
void umem_copy_from(void * const dest_ctx, uintptr_t dest_adr,
                    void * const src_ctx, uintptr_t src_adr,
                    size_t nbytes);
static inline 
void umem_copy_from_safe(void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                         void * const src_ctx, uintptr_t src_adr, size_t src_size,
                         size_t nbytes);

UMEM_EXTERN
uintptr_t umem_connect(void * const src_ctx, uintptr_t src_adr,
                       size_t nbytes, void * const dest_ctx,
                       size_t dest_alignment);
UMEM_EXTERN
void umem_sync_from(void * const src_ctx, uintptr_t src_adr,
                    void * const dest_ctx, uintptr_t dest_adr,
                    size_t nbytes);
UMEM_EXTERN
void umem_sync_to(void * const src_ctx, uintptr_t src_adr,
                  void * const dest_ctx, uintptr_t dest_adr,
                  size_t nbytes);
UMEM_EXTERN
void umem_disconnect(void * const src_ctx, uintptr_t src_adr,
                     void * const dest_ctx, uintptr_t dest_adr,
                     size_t dest_alignment);

UMEM_EXTERN
void umem_sync_to_safe(void * const src_ctx, uintptr_t src_adr, size_t src_size,
                       void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                       size_t nbytes);

UMEM_EXTERN
void umem_sync_from_safe(void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                         void * const src_ctx, uintptr_t src_adr, size_t src_size,
                         size_t nbytes);

/**
   Default implementations of memory context methods.
 */

/**
   umemVirtual destructor.
*/
UMEM_EXTERN
void umemVirtual_dtor(umemVirtual * const ctx);
/**
  umem_is_same_context returns true if the devices are the same in the
  sense of memory address spaces.
 */
UMEM_EXTERN
bool umemVirtual_is_same_context(umemVirtual * const one_ctx,
                                umemVirtual * const other_ctx);
UMEM_EXTERN
uintptr_t umemVirtual_calloc(umemVirtual * const ctx,
                             size_t nmemb, size_t size);
UMEM_EXTERN
uintptr_t umemVirtual_aligned_alloc(umemVirtual * const ctx,
                                    size_t alignment, size_t size);
/**
  umem_aligned_free frees device memory that was allocated using
  umem_aligned_alloc.
*/
UMEM_EXTERN
void umemVirtual_aligned_free(umemVirtual * const ctx, uintptr_t aligned_adr);
UMEM_EXTERN
uintptr_t umemVirtual_aligned_origin(umemVirtual * const ctx, uintptr_t aligned_adr);


/**
   Utility functions
*/

static inline
umemStatusType umem_get_status(void * const ctx);
static inline
const char * umem_get_message(void * const ctx);
static inline
bool umem_is_ok(void * const ctx);
UMEM_EXTERN
void umem_set_status(void * const ctx,
                     umemStatusType type, const char * message);
UMEM_EXTERN
void umem_clear_status(void * const ctx);
UMEM_EXTERN
const char* umem_get_status_name(umemStatusType type);
static inline
size_t umem_fundamental_align(void * const ctx);
UMEM_EXTERN
const char* umem_get_device_name_from_type(umemDeviceType type);
static inline
const char* umem_get_device_name(void * const ctx);
UMEM_EXTERN
void umem_copy_to_via_host(void * const src_ctx, uintptr_t src_adr,
                           void * const dest_ctx, uintptr_t dest_adr,
                           size_t nbytes);
UMEM_EXTERN
void umem_copy_from_via_host(void * const dest_ctx, uintptr_t dest_adr,
                             void * const src_ctx, uintptr_t src_adr,
                             size_t nbytes);

/**

   Implementations of inline functions

*/

static inline void umem_dtor(void * const ctx) {
  (*((umemVirtual * const)ctx)->vptr->dtor)((umemVirtual * const)ctx);
}

static inline uintptr_t umem_alloc(void * const ctx, size_t nbytes) {
  return (*((umemVirtual * const)ctx)->vptr->alloc)((umemVirtual * const)ctx, nbytes);
}

static inline uintptr_t umem_calloc(void * const ctx, size_t nmemb, size_t size) {
  return (*((umemVirtual * const)ctx)->vptr->calloc)((umemVirtual * const)ctx, nmemb, size);
}

static inline size_t umem_fundamental_align(void * const ctx) {
  switch (((umemVirtual * const)ctx)->type) {
  case umemFileDevice: return UMEM_FUNDAMENTAL_FILE_ALIGN;
  case umemCudaDevice: return UMEM_FUNDAMENTAL_CUDA_ALIGN;
  case umemHostDevice: return UMEM_FUNDAMENTAL_HOST_ALIGN;
  default: ;
  }
  return 1;
}

static inline bool umem_is_same_context(void * const one_ctx, void * const other_ctx) {
  return (one_ctx == other_ctx ? true :
          ((((umemVirtual * const)one_ctx)->type == ((umemVirtual * const)other_ctx)->type
            ? (*((umemVirtual * const)one_ctx)->vptr->is_same_context)((umemVirtual * const)one_ctx,
                                                                       (umemVirtual * const)other_ctx)
            : false)));
}

static inline umemStatusType umem_get_status(void * const ctx) {
  return ((umemVirtual * const)ctx)->status.type;
}

static inline const char * umem_get_message(void * const ctx) {
  return (((umemVirtual * const)ctx)->status.message == NULL ?
	  "" : ((umemVirtual * const)ctx)->status.message); }

static inline bool umem_is_ok(void * const ctx) {
  return ((umemVirtual * const)ctx)->status.type == umemOK;
}

static inline const char* umem_get_device_name(void * const ctx) {
  return umem_get_device_name_from_type(((umemVirtual * const)ctx)->type);
}

static inline uintptr_t umem_aligned_alloc(void * const ctx,
                                           size_t alignment, size_t size)
{
  TRY_RETURN(ctx, !umem_ispowerof2(alignment), umemValueError, return 0,
             "umem_aligned_alloc: alignment %zu must be power of 2",
             alignment);
  TRY_RETURN(ctx, size % alignment, umemValueError, return 0,
             "umem_aligned_alloc: size %zu must be multiple of alignment %zu",
             size, alignment);
  size_t fundamental_align = umem_fundamental_align(ctx);
  alignment = (alignment < fundamental_align ? fundamental_align : alignment);
  return (*((umemVirtual * const)ctx)->vptr->aligned_alloc)
    ((umemVirtual * const)ctx, alignment, size);
}

static inline uintptr_t umem_aligned_origin(void * const ctx,
                                            uintptr_t aligned_adr) {
  return (*((umemVirtual * const)ctx)->vptr->aligned_origin)
    ((umemVirtual * const)ctx, aligned_adr);
}

static inline void umem_free(void * const ctx, uintptr_t adr) {
  (*((umemVirtual * const)ctx)->vptr->free)((umemVirtual * const)ctx, adr);
}

static inline void umem_aligned_free(void * const ctx, uintptr_t aligned_adr) {
  (*((umemVirtual * const)ctx)->vptr->aligned_free)((umemVirtual * const)ctx, aligned_adr);
}

static inline void umem_copy_to(void * const src_ctx, uintptr_t src_adr,
				void * const dest_ctx, uintptr_t dest_adr,
				size_t nbytes) {
  (*((umemVirtual * const)src_ctx)->vptr->copy_to)((umemVirtual * const)src_ctx, src_adr,
                                                   (umemVirtual * const)dest_ctx, dest_adr, nbytes);
}

static inline void umem_copy_to_safe(void * const src_ctx, uintptr_t src_adr, size_t src_size,
                                     void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                                     size_t nbytes) {
  if (!(nbytes<=src_size))
    umem_set_status(src_ctx, umemIndexError, "umem_copy_to_safe: nbytes out of range of source context");
  else if (!(nbytes<=dest_size))
    umem_set_status(dest_ctx, umemIndexError, "umem_copy_to_safe: nbytes out of range of destination context");
  else
    umem_copy_to(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
}

static inline void umem_copy_from(void * const dest_ctx, uintptr_t dest_adr,
				  void * const src_ctx, uintptr_t src_adr,
				  size_t nbytes) {
  (*((umemVirtual * const)dest_ctx)->vptr->copy_from)((umemVirtual * const)dest_ctx, dest_adr,
                                                      (umemVirtual * const)src_ctx, src_adr, nbytes);
}

static inline void umem_copy_from_safe(void * const dest_ctx, uintptr_t dest_adr, size_t dest_size,
                                       void * const src_ctx, uintptr_t src_adr, size_t src_size,
                                       size_t nbytes) {
  if (!(nbytes<=src_size))
    umem_set_status(src_ctx, umemIndexError, "umem_copy_from_safe: nbytes out of range of source context");
  else if (!(nbytes<=dest_size))
    umem_set_status(dest_ctx, umemIndexError, "umem_copy_from_safe: nbytes out of range of destination context");
  else
    umem_copy_from(dest_ctx, dest_adr, src_ctx, src_adr, nbytes);
}

static inline void umem_set(void * const ctx, uintptr_t adr, int c, size_t nbytes) {
  (*((umemVirtual * const)ctx)->vptr->set)((umemVirtual * const)ctx, adr, c, nbytes);
}

static inline void umem_set_safe(void * const ctx, uintptr_t adr, size_t size, int c, size_t nbytes) {
  if (size < nbytes)
    umem_set_status(ctx, umemIndexError, "umem_set_safe: nbytes out of range of context");
  else
    umem_set(ctx, adr, c, nbytes);
}


CLOSE_EXTERN_C

#ifdef __cplusplus
#include "umem.hpp"
#endif

#endif
