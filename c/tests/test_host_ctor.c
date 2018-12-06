#include "umem_testing.h"

int main() {
  umemHost host;
  assert_eq((void*)&host, (void*)&host.super);
  umemHost_ctor(&host);
  assert_is_ok(host);
  umem_clear_status(&host.super);
  assert_is_ok(host);
  umem_dtor(&host);
  RETURN_STATUS;
}
