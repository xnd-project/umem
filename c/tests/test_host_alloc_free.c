#include "umem_testing.h"

int main() {
  umemHost host;
  umemHost_ctor(&host);
  uintptr_t addr = umem_alloc(&host, 10);
  assert_is_ok(host);
  umem_free(&host, addr);
  assert_is_ok(host);
  umem_dtor(&host);
  RETURN_STATUS;
}
