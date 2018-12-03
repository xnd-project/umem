#ifndef UMEM_TESTING_H
#define UMEM_TESTING_H

#include <assert.h>
#include <errno.h>
#include <string.h>
#include "umem.h"

// portable err(eval, fmt, ...) from err.h
#define ERR(EVAL, FMT, ...)			\
  do {						\
    fprintf(stderr, FMT, __VA_ARGS__);		\
    return EVAL;				\
  } while(0)

#define RETURN_STATUS							\
  do {									\
    ERR((errno ? EXIT_FAILURE : EXIT_SUCCESS), "(%s#%d)", __FILE__, __LINE__); \
  } while (0)

#define assert_str_eq(lhs, rhs)						\
  do {									\
    if (strcmp(lhs, rhs) != 0) {\
      errno = ECANCELED;						\
      ERR(EXIT_FAILURE, "assert(\"%s\" == \"%s\") FAILED (%s#%d)",	\
	  lhs, rhs,							\
	  __FILE__, __LINE__);						\
    }									\
  } while(0)

#define assert_is_ok(dev)						\
  do {									\
    if (!umem_is_ok(&dev)) {						\
      errno = ECANCELED;						\
      umemDeviceType status = umem_get_status(&dev);			\
      ERR(EXIT_FAILURE, "%s: %s (%s#%d)" , umem_get_status_name(status), \
	  umem_get_message(&dev), __FILE__, __LINE__);			\
    }									\
  } while(0)

#define assert_is_not_ok(dev)						\
  do {									\
    if (umem_is_ok(&dev)) {						\
      errno = ECANCELED;						\
      umemDeviceType status = umem_get_status(&dev);			\
      ERR(EXIT_FAILURE, "%s: %s (%s#%d)" , umem_get_status_name(status), \
	  umem_get_message(&dev), __FILE__, __LINE__);			\
    }									\
  } while(0)

#endif
