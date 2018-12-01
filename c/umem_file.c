#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include "umem.h"

#define FILE_CALL(ME, CALL, ERROR, ERRRETURN, FMT, ...)		\
  do { assert(!errno);						\
    int status = (CALL);					\
  if (status != 0) {						\
    char buf[256];						\
    sprintf(buf, FMT " -> %d [errno=%d (%s)]", __VA_ARGS__,	\
	    status, errno, strerror(errno));			\
    umem_set_status(ME, ERROR, buf);				\
    ERRRETURN;							\
      } assert(!errno);						\
 } while (0)


/*
  Implementations of umemFile methods.
*/

static uintptr_t umemFile_alloc_(umemVirtual * const me, size_t nbytes) {
  assert(me->type == umemFileDevice);
  umemFile * const  me_ = (umemFile * const)me;
  if (me_->fp == 0) {
    FILE_CALL(me, !(me_->fp = (uintptr_t)fopen(me_->filename, me_->mode)),
	      umemIOError, return -1,
	      "umemFile_alloc_: !fopen(\"%s\", \"%s\")",
	      me_->filename, me_->mode);
    return 0;
  }
  long pos = -1;
  FILE_CALL(me, !((pos = ftell((FILE*)me_->fp)) == -1),
	    umemIOError, return -1,
	    "umemFile_alloc_: !(ftell(%lx)==-1)", me_->fp);
  return pos;
}


static void umemFile_free_(umemVirtual * const me, uintptr_t adr) {
  assert(me->type == umemFileDevice);
  umemFile * const me_ = (umemFile * const)me;
  if (me_->fp) {
    FILE_CALL(me, fclose((FILE*)me_->fp), umemIOError, return,
	      "umemFile_free_: fclose(%lx)", me_->fp);
    me_->fp = 0;
  }
}


static void umemFile_dtor_(umemVirtual * const me) {
  umemFile_free_(me, 0);    
  umemVirtual_dtor(me);
}


static void umemFile_set_(umemVirtual * const me,
			  uintptr_t adr, int c, size_t nbytes) {
  assert(me->type == umemFileDevice);
  umemFile * const me_ = (umemFile * const)me;
  char cbuf[4092];
  size_t bbytes = (nbytes > sizeof(cbuf) ? sizeof(cbuf) : nbytes);
  memset(cbuf, c, bbytes);
  size_t bytes = nbytes;
  size_t wbytes = 0;
  while (bytes > 0) {
    bbytes = (bytes > sizeof(cbuf) ? sizeof(cbuf) : bytes);
    wbytes += fwrite(cbuf, 1, bbytes,  (FILE *)me_->fp);
    FILE_CALL(me, ferror((FILE*)me_->fp), umemIOError,
	      do { clearerr((FILE*)me_->fp); return;} while(0),
	      "umemFile_set_: fwrite(%p, 1, %ld, %lx)",
	      cbuf, bbytes, me_->fp);
    bytes -= bbytes;
  }
  assert (nbytes == wbytes);
}


static void umemFile_copy_to_(umemVirtual * const me, uintptr_t src_adr,
			      umemVirtual * const she, uintptr_t dest_adr,
			      size_t nbytes) {
  assert(me->type == umemFileDevice);
  umemFile * const me_ = (umemFile * const)me;
  assert(me_->fp != 0);
  switch(she->type) {
  case umemHostDevice:
    FILE_CALL(me, (fseek((FILE*)me_->fp, src_adr, SEEK_SET) == -1),
	      umemIOError, return,
	      "umemFile_copy_to_: (fseek(%lx, %lx, SEEK_SET)==-1)",
	      me_->fp, src_adr);
    size_t rbytes;
    FILE_CALL(me, !((rbytes=fread((void *)dest_adr, 1,
				  nbytes, (FILE*)me_->fp))==nbytes),
	      umemIOError, return,
	      "umemFile_copy_to_: fread(%lx, 1, %ld, %lx)==%ld!=%ld",
	      dest_adr, nbytes, me_->fp, rbytes, nbytes);
    FILE_CALL(me, ferror((FILE*)me_->fp), umemIOError, return,
	      "umemFile_copy_to_: fread(%lx, 1, %ld, %lx)",
	      dest_adr, nbytes, me_->fp);
    break;
  case umemFileDevice:
    //TODO: write to another file
    {
      char buf[256];
      sprintf(buf, "umemFile_copy_to_(%p, %lx, %p, %lx, %ld)",
	      me, src_adr, she, dest_adr, nbytes);
      umem_set_status(me, umemNotImplementedError, buf);
    }
    break;
  default:
    umem_copy_from(she, dest_adr, me, src_adr, nbytes);
  }
}


static void umemFile_copy_from_(umemVirtual * const me, uintptr_t dest_adr,
				umemVirtual * const she, uintptr_t src_adr,
				size_t nbytes) {
  assert(me->type == umemFileDevice);
  umemFile * const me_ = (umemFile * const)me;
  assert(me_->fp != 0);
  switch(she->type) {
  case umemHostDevice:
    FILE_CALL(me, (fseek((FILE*)me_->fp, dest_adr, SEEK_SET) == -1),
	      umemIOError, return,
	      "umemFile_copy_from_: (fseek(%lx, %lx, SEEK_SET)==-1)",
	      me_->fp, dest_adr);
    size_t wbytes;
    FILE_CALL(me, !((wbytes = fwrite((const void *)src_adr, 1,
				     nbytes, (FILE *)me_->fp))==nbytes),
	      umemIOError, return,
	      "umemFile_copy_from_: fwrite(%lx, 1, %ld, %lx)==%ld!=%ld",
	      src_adr, nbytes, me_->fp, wbytes, nbytes);
    break;
  case umemFileDevice:
    //TODO: read from another file
    break;
  default:
    umem_copy_to(she, src_adr, me, dest_adr, nbytes);
  }
}


/*
  umemFile constructor.
*/
void umemFile_ctor(umemFile * const me,
		   const char * filename, const char * mode) {
  static struct umemVtbl const vtbl = {
    &umemFile_dtor_,
    &umemFile_alloc_,
    &umemVirtual_calloc,
    &umemFile_free_,
    &umemFile_set_,
    &umemFile_copy_to_,
    &umemFile_copy_from_,
  };
  assert(me==(umemFile * const)&me->super);
  umemVirtual_ctor(&me->super);
  me->super.type = umemFileDevice;
  me->super.vptr = &vtbl;
  me->fp = 0;
  me->filename = filename;
  me->mode = mode;
}


