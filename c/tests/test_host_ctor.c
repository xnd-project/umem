#include "umem_testing.h"

int main() {
  umemHost host;
  assert(((void*)&host) == ((void*)&host.super));
  umemHost_ctor(&host);
  assert(umem_is_ok(&host.super));
  umem_clear_status(&host.super);
  assert(umem_is_ok(&host.super));
  umemHost_dtor(&host);
  RETURN_STATUS;
}
