#include "umem_testing.h"

int main() {
  {
    umem::Host host;
    assert_is_ok(host);
    umem::Address adr = host.alloc(11);
    assert_is_ok(host);
    adr.set(97, 5);
    assert_is_ok(host);
    (adr+5).set(98, 5);
    assert_is_ok(host);
    adr[10] = 0;
    assert_is_ok(host);
    assert_nstr_eq(10, (char*)adr, "aaaaabbbbb");
    assert_is_ok(host);
  }
  RETURN_STATUS;
}
