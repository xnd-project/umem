#include "umem_testing.h"

int main() {
  umemHost host;
  umemHost_ctor(&host);
  uintptr_t addr = umem_alloc(&host, 10);
  assert(umem_is_ok(&host.super));
  umem_free(&host, addr);
  assert(umem_is_ok(&host.super));
  umemHost_dtor(&host);
  RETURN_STATUS;
}
