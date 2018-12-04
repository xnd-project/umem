#include "umem_testing.h"

int main() {
  umemHost host;
  umemHost_ctor(&host);
  uintptr_t adr = umem_calloc(&host, 16, 10);
  assert_is_ok(host);
  umem_free(&host, adr);
  assert_is_ok(host);
  umem_dtor(&host);
  RETURN_STATUS;
}
