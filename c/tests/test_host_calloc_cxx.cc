#include "umem_testing.h"

int main() {
  {
    umem::Host host;
    umem::Address adr = host.calloc(16, 10);
    assert_is_ok(host);
  }
  RETURN_STATUS;
}
