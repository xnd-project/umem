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

#define FILE_CALL(CTX, CALL, ERROR, ERRRETURN, FMT, ...)			\
  do { assert(!errno);                                              \
  int status = (CALL);							\
  if (status != 0 || errno) {							\
    char buf[256];							\
    ERRBUF								\
    snprintf(buf, sizeof(buf), FMT " -> %d [errno=%d (%s)]", __VA_ARGS__, \
	     status, errno, errbuf);				\
    umem_set_status(CTX, ERROR, buf);					\
    ERRRETURN;								\
  } assert(!errno);							\
  } while (0)


/*
  Implementations of umemFile methods.
*/

static uintptr_t umemFile_alloc_(umemVirtual * const ctx, size_t nbytes) {
  assert(ctx->type == umemFileDevice);
  umemFile * const  ctx_ = (umemFile * const)ctx;
  if (ctx_->fp == 0) {
#ifdef _MSC_VER
    FILE_CALL(ctx, fopen_s(&((FILE*)ctx_->fp), ctx_->filename, ctx_->mode),
	      umemIOError, return -1,
	      "umemFile_alloc_: !fopen_s(&fp, \"%s\", \"%s\")",
	      ctx_->filename, ctx_->mode);
#else
    FILE_CALL(ctx, !(ctx_->fp = (uintptr_t)fopen(ctx_->filename, ctx_->mode)),
	      umemIOError, return -1,
	      "umemFile_alloc_: !fopen(\"%s\", \"%s\")",
	      ctx_->filename, ctx_->mode);
#endif
    return 0;
  }
  long pos = -1;
  FILE_CALL(ctx, !((pos = ftell((FILE*)ctx_->fp)) == -1),
	    umemIOError, return -1,
	    "umemFile_alloc_: !(ftell(%" PRIxPTR ")==-1)", ctx_->fp);
  return pos;
}


static void umemFile_free_(umemVirtual * const ctx, uintptr_t adr) {
  assert(ctx->type == umemFileDevice);
  umemFile * const ctx_ = (umemFile * const)ctx;
  if (ctx_->fp) {
    FILE_CALL(ctx, fclose((FILE*)ctx_->fp), umemIOError, return,
	      "umemFile_free_: fclose(%" PRIxPTR ")", ctx_->fp);
    ctx_->fp = 0;
  }
}


static void umemFile_dtor_(umemVirtual * const ctx) {
  umemFile * const ctx_ = (umemFile * const)ctx;
  umemFile_free_(ctx, 0);
  umem_free(ctx->host_ctx, (uintptr_t)ctx_->mode);
  umem_free(ctx->host_ctx, (uintptr_t)ctx_->filename);
  umemVirtual_dtor(ctx);
}


static void umemFile_set_(umemVirtual * const ctx,
			  uintptr_t adr, int c, size_t nbytes) {
  assert(ctx->type == umemFileDevice);
  umemFile * const ctx_ = (umemFile * const)ctx;
  char cbuf[4092];
  size_t bbytes = (nbytes > sizeof(cbuf) ? sizeof(cbuf) : nbytes);
  memset(cbuf, c, bbytes);
  size_t bytes = nbytes;
  size_t wbytes = 0;
  while (bytes > 0) {
    bbytes = (bytes > sizeof(cbuf) ? sizeof(cbuf) : bytes);
    wbytes += fwrite(cbuf, 1, bbytes,  (FILE *)ctx_->fp);
    FILE_CALL(ctx, ferror((FILE*)ctx_->fp), umemIOError,
	      do { clearerr((FILE*)ctx_->fp); return;} while(0),
	      "umemFile_set_: fwrite(%p, 1, %zu, %" PRIxPTR ")",
	      cbuf, bbytes, ctx_->fp);
    bytes -= bbytes;
  }
  assert (nbytes == wbytes);
}


static void umemFile_copy_to_(umemVirtual * const src_ctx, uintptr_t src_adr,
			      umemVirtual * const dest_ctx, uintptr_t dest_adr,
			      size_t nbytes) {
  assert(src_ctx->type == umemFileDevice);
  umemFile * const src_ctx_ = (umemFile * const)src_ctx;
  assert(src_ctx_->fp != 0);
  switch(dest_ctx->type) {
  case umemHostDevice:
    FILE_CALL(src_ctx, (fseek((FILE*)src_ctx_->fp, (long)src_adr, SEEK_SET) == -1),
	      umemIOError, return,
	      "umemFile_copy_to_: (fseek(%" PRIxPTR ", %" PRIxPTR ", SEEK_SET)==-1)",
	      src_ctx_->fp, src_adr);
    size_t rbytes;
    FILE_CALL(src_ctx, !((rbytes=fread((void *)dest_adr, 1,
				  nbytes, (FILE*)src_ctx_->fp))==nbytes),
	      umemIOError, return,
	      "umemFile_copy_to_: fread(%" PRIxPTR ", 1, %zu, %" PRIxPTR ")==%zu!=%zu",
	      dest_adr, nbytes, src_ctx_->fp, rbytes, nbytes);
    FILE_CALL(src_ctx, ferror((FILE*)src_ctx_->fp), umemIOError, return,
	      "umemFile_copy_to_: fread(%" PRIxPTR ", 1, %zu, %" PRIxPTR ")",
	      dest_adr, nbytes, src_ctx_->fp);
    break;
  case umemFileDevice:
    //TODO: write to another file
    {
      char buf[256];
      snprintf(buf, sizeof(buf), "umemFile_copy_to_(%p, %" PRIxPTR ", %p, %" PRIxPTR ", %zu)",
	       (void*)src_ctx, src_adr, (void*)dest_ctx, dest_adr, nbytes);
      umem_set_status(src_ctx, umemNotImplementedError, buf);
    }
    break;
  default:
    umem_copy_from(dest_ctx, dest_adr, src_ctx, src_adr, nbytes);
  }
}


static void umemFile_copy_from_(umemVirtual * const dest_ctx, uintptr_t dest_adr,
				umemVirtual * const src_ctx, uintptr_t src_adr,
				size_t nbytes) {
  assert(dest_ctx->type == umemFileDevice);
  umemFile * const dest_ctx_ = (umemFile * const)dest_ctx;
  assert(dest_ctx_->fp != 0);
  switch(src_ctx->type) {
  case umemHostDevice:
    FILE_CALL(dest_ctx, (fseek((FILE*)dest_ctx_->fp, (long)dest_adr, SEEK_SET) == -1),
	      umemIOError, return,
	      "umemFile_copy_from_: (fseek(%" PRIxPTR ", %" PRIxPTR ", SEEK_SET)==-1)",
	      dest_ctx_->fp, dest_adr);
    size_t wbytes;
    FILE_CALL(dest_ctx, !((wbytes = fwrite((const void *)src_adr, 1,
				     nbytes, (FILE *)dest_ctx_->fp))==nbytes),
	      umemIOError, return,
	      "umemFile_copy_from_: fwrite(%" PRIxPTR ", 1, %zu, %" PRIxPTR ")==%zu!=%zu",
	      src_adr, nbytes, dest_ctx_->fp, wbytes, nbytes);
    break;
  case umemFileDevice:
    //TODO: read from another file
    break;
  default:
    umem_copy_to(src_ctx, src_adr, dest_ctx, dest_adr, nbytes);
  }
}

bool umemFile_is_accessible_from_(umemVirtual * const one_ctx, umemVirtual * const other_ctx) {
  assert(one_ctx->type == umemFileDevice);
  if (other_ctx->type == umemFileDevice)
    return (strcmp(((umemFile * const)one_ctx)->filename, ((umemFile * const)other_ctx)->filename) == 0 ? true : false);
  return false;
}

/*
  umemFile constructor.
*/
void umemFile_ctor(umemFile * const ctx,
		   const char * filename, const char * mode) {
  static struct umemVtbl const vtbl = {
    &umemFile_dtor_,
    &umemFile_is_accessible_from_,
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
  assert(ctx==(umemFile * const)&ctx->super);
  umemHost_ctor(&ctx->host);
  umemVirtual_ctor(&ctx->super, &ctx->host);
  ctx->super.type = umemFileDevice;
  ctx->super.vptr = &vtbl;
  ctx->fp = 0;
  if (strlen(filename) == 0)
    umem_set_status(ctx, umemValueError, "invalid 0-length filename");
  if (strlen(mode) == 0)
    umem_set_status(ctx, umemValueError, "invalid 0-length mode");

  if (umem_is_ok(ctx)) {
    size_t l = strlen(filename);
    ctx->filename = (const char*)umem_calloc(&ctx->host, l+1, 1);
    umem_copy_from(&ctx->host, (uintptr_t)ctx->filename, &ctx->host, (uintptr_t)filename, l);
    assert(umem_is_ok(&ctx->host));

    l = strlen(mode);
    ctx->mode = (const char*)umem_calloc(&ctx->host, l+1, 1);
    umem_copy_from(&ctx->host, (uintptr_t)ctx->mode, &ctx->host, (uintptr_t)mode, l);
    assert(umem_is_ok(&ctx->host));

  } else {
    ctx->filename = NULL;
    ctx->mode = NULL;
  }
}


