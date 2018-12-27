#ifndef UMEM_HOST_UTILS_H
#define UMEM_HOST_UTILS_H

#define HOST_CALL(CTX, CALL, ERROR, ERRRETURN, FMT, ...)       \
  do {								\
    if (CALL) {				\
      char buf[256];						\
      snprintf(buf, sizeof(buf), FMT, __VA_ARGS__);             \
      umem_set_status(CTX, ERROR, buf);                        \
      ERRRETURN;						\
    }								\
  } while (0)

#endif
