#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include "umem.h"

#ifdef _MSC_VER
#define ERRBUF					\
  char errbuf[96];				\
  strerror_s(errbuf, 96, errno);
#else
#define ERRBUF					\
  char errbuf[96];				\
  strerror_r(errno, errbuf, 96);
#endif

#define FILE_CALL(THIS, CALL, ERROR, ERRRETURN, FMT, ...)			\
  do { assert(!errno);							\
  int status = (CALL);							\
  if (status != 0) {							\
    char buf[256];							\
    ERRBUF								\
    snprintf(buf, sizeof(buf), FMT " -> %d [errno=%d (%s)]", __VA_ARGS__, \
	     status, errno, errbuf);				\
    umem_set_status(THIS, ERROR, buf);					\
    ERRRETURN;								\
  } assert(!errno);							\
  } while (0)


/*
  Implementations of umemFile methods.
*/

static uintptr_t umemFile_alloc_(umemVirtual * const this, size_t nbytes) {
  assert(this->type == umemFileDevice);
  umemFile * const  this_ = (umemFile * const)this;
  if (this_->fp == 0) {
#ifdef _MSC_VER
    FILE_CALL(this, fopen_s(&((FILE*)this_->fp), this_->filename, this_->mode),
	      umemIOError, return -1,
	      "umemFile_alloc_: !fopen_s(&fp, \"%s\", \"%s\")",
	      this_->filename, this_->mode);
#else
    FILE_CALL(this, !(this_->fp = (uintptr_t)fopen(this_->filename, this_->mode)),
	      umemIOError, return -1,
	      "umemFile_alloc_: !fopen(\"%s\", \"%s\")",
	      this_->filename, this_->mode);
#endif
    return 0;
  }
  long pos = -1;
  FILE_CALL(this, !((pos = ftell((FILE*)this_->fp)) == -1),
	    umemIOError, return -1,
	    "umemFile_alloc_: !(ftell(%" PRIxPTR ")==-1)", this_->fp);
  return pos;
}


static void umemFile_free_(umemVirtual * const this, uintptr_t adr) {
  assert(this->type == umemFileDevice);
  umemFile * const this_ = (umemFile * const)this;
  if (this_->fp) {
    FILE_CALL(this, fclose((FILE*)this_->fp), umemIOError, return,
	      "umemFile_free_: fclose(%" PRIxPTR ")", this_->fp);
    this_->fp = 0;
  }
}


static void umemFile_dtor_(umemVirtual * const this) {
  umemFile_free_(this, 0);
  umemVirtual_dtor(this);
}


static void umemFile_set_(umemVirtual * const this,
			  uintptr_t adr, int c, size_t nbytes) {
  assert(this->type == umemFileDevice);
  umemFile * const this_ = (umemFile * const)this;
  char cbuf[4092];
  size_t bbytes = (nbytes > sizeof(cbuf) ? sizeof(cbuf) : nbytes);
  memset(cbuf, c, bbytes);
  size_t bytes = nbytes;
  size_t wbytes = 0;
  while (bytes > 0) {
    bbytes = (bytes > sizeof(cbuf) ? sizeof(cbuf) : bytes);
    wbytes += fwrite(cbuf, 1, bbytes,  (FILE *)this_->fp);
    FILE_CALL(this, ferror((FILE*)this_->fp), umemIOError,
	      do { clearerr((FILE*)this_->fp); return;} while(0),
	      "umemFile_set_: fwrite(%p, 1, %zu, %" PRIxPTR ")",
	      cbuf, bbytes, this_->fp);
    bytes -= bbytes;
  }
  assert (nbytes == wbytes);
}


static void umemFile_copy_to_(umemVirtual * const this, uintptr_t src_adr,
			      umemVirtual * const that, uintptr_t dest_adr,
			      size_t nbytes) {
  assert(this->type == umemFileDevice);
  umemFile * const this_ = (umemFile * const)this;
  assert(this_->fp != 0);
  switch(that->type) {
  case umemHostDevice:
    FILE_CALL(this, (fseek((FILE*)this_->fp, (long)src_adr, SEEK_SET) == -1),
	      umemIOError, return,
	      "umemFile_copy_to_: (fseek(%" PRIxPTR ", %" PRIxPTR ", SEEK_SET)==-1)",
	      this_->fp, src_adr);
    size_t rbytes;
    FILE_CALL(this, !((rbytes=fread((void *)dest_adr, 1,
				  nbytes, (FILE*)this_->fp))==nbytes),
	      umemIOError, return,
	      "umemFile_copy_to_: fread(%" PRIxPTR ", 1, %zu, %" PRIxPTR ")==%zu!=%zu",
	      dest_adr, nbytes, this_->fp, rbytes, nbytes);
    FILE_CALL(this, ferror((FILE*)this_->fp), umemIOError, return,
	      "umemFile_copy_to_: fread(%" PRIxPTR ", 1, %zu, %" PRIxPTR ")",
	      dest_adr, nbytes, this_->fp);
    break;
  case umemFileDevice:
    //TODO: write to another file
    {
      char buf[256];
      snprintf(buf, sizeof(buf), "umemFile_copy_to_(%p, %" PRIxPTR ", %p, %" PRIxPTR ", %zu)",
	       this, src_adr, that, dest_adr, nbytes);
      umem_set_status(this, umemNotImplementedError, buf);
    }
    break;
  default:
    umem_copy_from(that, dest_adr, this, src_adr, nbytes);
  }
}


static void umemFile_copy_from_(umemVirtual * const this, uintptr_t dest_adr,
				umemVirtual * const that, uintptr_t src_adr,
				size_t nbytes) {
  assert(this->type == umemFileDevice);
  umemFile * const this_ = (umemFile * const)this;
  assert(this_->fp != 0);
  switch(that->type) {
  case umemHostDevice:
    FILE_CALL(this, (fseek((FILE*)this_->fp, (long)dest_adr, SEEK_SET) == -1),
	      umemIOError, return,
	      "umemFile_copy_from_: (fseek(%" PRIxPTR ", %" PRIxPTR ", SEEK_SET)==-1)",
	      this_->fp, dest_adr);
    size_t wbytes;
    FILE_CALL(this, !((wbytes = fwrite((const void *)src_adr, 1,
				     nbytes, (FILE *)this_->fp))==nbytes),
	      umemIOError, return,
	      "umemFile_copy_from_: fwrite(%" PRIxPTR ", 1, %zu, %" PRIxPTR ")==%zu!=%zu",
	      src_adr, nbytes, this_->fp, wbytes, nbytes);
    break;
  case umemFileDevice:
    //TODO: read from another file
    break;
  default:
    umem_copy_to(that, src_adr, this, dest_adr, nbytes);
  }
}


bool umemFile_is_same_device_(umemVirtual * const this, umemVirtual * const that) {
  umemFile * const this_ = (umemFile * const)this;
  umemFile * const that_ = (umemFile * const)that;
  return (strcmp(this_->filename, that_->filename) == 0 ? true : false);
}

/*
  umemFile constructor.
*/
void umemFile_ctor(umemFile * const this,
		   const char * filename, const char * mode) {
  static struct umemVtbl const vtbl = {
    &umemFile_dtor_,
    &umemFile_is_same_device_,
    &umemFile_alloc_,
    &umemVirtual_calloc,
    &umemFile_free_,
    &umemVirtual_aligned_alloc,
    &umemVirtual_aligned_origin,
    &umemVirtual_aligned_free,
    &umemFile_set_,
    &umemFile_copy_to_,
    &umemFile_copy_from_,
  };
  assert(this==(umemFile * const)&this->super);
  umemHost_ctor(&this->host);
  umemVirtual_ctor(&this->super, &this->host);
  this->super.type = umemFileDevice;
  this->super.vptr = &vtbl;
  this->fp = 0;
  this->filename = filename;
  this->mode = mode;
}


