#ifndef XNDMemory_XNDMEMORY_H
#define XNDMemory_XNDMEMORY_H
/*
  Author: Pearu Peterson
  Created: November 2018
*/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

/* umemDeviceType defines the flags of supported device memory
   classes.
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
*/
struct umemVtbl;
typedef struct {
  struct umemVtbl const *vptr;
  umemDeviceType type;
  umemStatus status;
} umemVirtual;

extern void umemVirtual_ctor(umemVirtual * const me); /* Constructor */
extern void umemVirtual_dtor(umemVirtual * const me); /* Destructor */

/*
  Status and error handling functions.
*/

static inline umemStatusType umem_get_status(umemVirtual * const me) {
  return me->status.type;
}
static inline const char * umem_get_message(umemVirtual * const me) {
  static const char empty[] = "";
  return (me->status.message == NULL ? empty : me->status.message); }
static inline int umem_is_ok(umemVirtual * const me) {
  return me->status.type == umemOK;
}
extern void umem_set_status(umemVirtual * const me,
			    umemStatusType type, const char * message);
extern void umem_clear_status(umemVirtual * const me);

/*
  umemHost represents host memory.
*/

typedef struct {
  umemVirtual super;
} umemHost;

extern void umemHost_ctor(umemHost * const me);  /* Constructor */
static inline void umemHost_dtor(umemHost * const me) {
  umemVirtual_dtor(&me->super);
}

/*
  umemVtbl defines a table of umemVirtual methods.
*/

struct umemVtbl {
  uintptr_t (*alloc)(umemVirtual * const me, size_t nbytes);
  void (*free)(umemVirtual * const me, uintptr_t adr);
  void (*set)(umemVirtual * const me, uintptr_t adr, int c, size_t nbytes);
  void (*copy_to)(umemVirtual * const me, uintptr_t src_adr,
		  umemVirtual * const she, uintptr_t dest_adr,
		  size_t nbytes);
  void (*copy_from)(umemVirtual * const me, uintptr_t dest_adr,
		    umemVirtual * const she, uintptr_t src_adr,
		    size_t nbytes);
};

/*
  umemCuda represents GPU device memory using CUDA library.
*/
#ifdef HAVE_CUDA
typedef struct {
  umemVirtual super;
  int device;
} umemCuda;

extern void umemCuda_ctor(umemCuda * const me, int device); /* Constructor */
static inline void umemCuda_dtor(umemCuda * const me) {
  umemVirtual_dtor(&me->super);
}
#endif

/*
  umemFile represents FILE in binary format.

  alloc - opens the file with given mode, nbytes argument is ignored
  free  - closes the file
  adr   - is the file position relative to the start of the file
*/
typedef struct {
  umemVirtual super;
  uintptr_t fp;
  const char * filename;
  const char * mode;
} umemFile;

extern void umemFile_ctor(umemFile * const me,
			  const char* filename, const char* mode);  /* Constructor */
extern void umemFile_dtor(umemFile * const me);  /* Destructor */

/* Public API */

/*
  umem_alloc allocates device memory and returns the memory addresss.
 */
static inline uintptr_t umem_alloc(void * const me, size_t nbytes) {
  uintptr_t adr = (*((umemVirtual * const)me)->vptr->alloc)(me, nbytes);
  return adr;
}

/*
  umem_free frees device memory that was allocated using umem_alloc.
 */
static inline void umem_free(void * const me, uintptr_t adr) {
  (*((umemVirtual * const)me)->vptr->free)(me, adr);
}

/*
  umem_set fills memory with constant byte
*/
static inline void umem_set(void * const me, uintptr_t adr, int c, size_t nbytes) {
  (*((umemVirtual * const)me)->vptr->set)(me, adr, c, nbytes);
}

/*
  umem_copy_from and umem_copy_to copy data from one device to another.
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
  Internal methods for copying data from one device to another using
  host memory as a buffer.
 */
extern void umem_copy_to_via_host(void * const me, uintptr_t src_adr,
				  void * const she, uintptr_t dest_adr,
				  size_t nbytes);
extern void umem_copy_from_via_host(void * const me, uintptr_t dest_adr,
				    void * const she, uintptr_t src_adr,
				    size_t nbytes);

/*
  Various utility functions.
*/
extern const char* umem_get_device_name(umemDeviceType type);
extern const char* umem_get_status_name(umemStatusType type);
#endif
