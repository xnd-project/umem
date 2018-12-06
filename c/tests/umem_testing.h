#ifndef UMEM_TESTING_H
#define UMEM_TESTING_H

#include <assert.h>
#include <errno.h>
#include <string.h>
#include "umem.h"

// portable err(eval, fmt, ...) from err.h
#define ERR(EVAL, FMT, ...)			\
  do {						\
    fprintf(stderr, FMT "\n", __VA_ARGS__);     \
    return EVAL;				\
  } while(0)

#define RETURN_STATUS							\
  do {									\
    ERR((errno ? EXIT_FAILURE : EXIT_SUCCESS),                          \
        "%s (%s#%d)",                                                   \
        (errno ? "FAILURE" : "SUCCESS"),                                \
        __FILE__, __LINE__);                                            \
  } while (0)

#define STR(X) #X

#define assert_eq(LHS, RHS)						\
  do {									\
    if ((LHS) != (RHS)) {                                               \
      errno = ECANCELED;						\
      ERR(EXIT_FAILURE, "assert( " STR(LHS) " == " STR(RHS) ") FAILED (%s#%d)", \
	  __FILE__, __LINE__);						\
    }									\
  } while(0)

#define assert_int_eq(LHS, RHS)						\
  do {									\
    if ((LHS) != (RHS)) {                                               \
      errno = ECANCELED;						\
      ERR(EXIT_FAILURE, "assert( (" STR(LHS) ")->%d == (" STR(RHS) ")->%d) FAILED (%s#%d)", \
          LHS, RHS,                                                     \
	  __FILE__, __LINE__);						\
    }									\
  } while(0)

#define assert_int_ne(LHS, RHS)						\
  do {									\
    if ((LHS) == (RHS)) {                                               \
      errno = ECANCELED;						\
      ERR(EXIT_FAILURE, "assert( (" STR(LHS) ")->%d != (" STR(RHS) ")->%d) FAILED (%s#%d)", \
          LHS, RHS,                                                     \
	  __FILE__, __LINE__);						\
    }									\
  } while(0)

#define assert_str_eq(LHS, RHS)						\
  do {									\
    if (strlen(LHS)!=strlen(RHS) || strcmp(LHS, RHS) != 0) {            \
      errno = ECANCELED;						\
      ERR(EXIT_FAILURE, "assert(\"%s\" == \"%s\") FAILED (%s#%d)",	\
	  LHS, RHS,							\
	  __FILE__, __LINE__);						\
    }									\
  } while(0)

#define assert_nstr_eq(N, LHS, RHS)                                     \
  do {									\
    if (strncmp(LHS, RHS, N) != 0) {                                    \
      errno = ECANCELED;						\
      ERR(EXIT_FAILURE, "assert(\"%s\" == \"%s\") FAILED (%s#%d)",	\
	  LHS, RHS,							\
	  __FILE__, __LINE__);						\
    }									\
  } while(0)

#define assert_is_ok(DEV)						\
  do {									\
    if (!umem_is_ok(&DEV)) {						\
      errno = ECANCELED;						\
      umemDeviceType status = umem_get_status(&DEV);			\
      ERR(EXIT_FAILURE, "%s: %s (%s#%d)" , umem_get_status_name(status), \
	  umem_get_message(&DEV), __FILE__, __LINE__);			\
    }									\
  } while(0)

#define assert_is_not_ok(DEV)						\
  do {									\
    if (umem_is_ok(&DEV)) {						\
      errno = ECANCELED;						\
      umemDeviceType status = umem_get_status(&DEV);			\
      ERR(EXIT_FAILURE, "%s: %s (%s#%d)" , umem_get_status_name(status), \
	  umem_get_message(&DEV), __FILE__, __LINE__);			\
    }									\
  } while(0)

#endif
