

#define CUDA_CALL(CTX, CALL, ERROR, ERRRETURN, FMT, ...)                \
  do {									\
    int old_errno = errno;						\
    cudaError_t error = CALL;						\
    if (error != cudaSuccess) {						\
      char buf[256];							\
      snprintf(buf, sizeof(buf), FMT " -> %s: %s", __VA_ARGS__,		\
	       cudaGetErrorName(error), cudaGetErrorString(error));	\
      umem_set_status(CTX, ERROR, buf);					\
      ERRRETURN;							\
    } else errno = old_errno;						\
  } while (0)
