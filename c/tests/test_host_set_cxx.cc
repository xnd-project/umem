#include "umem_testing.h"

int main() {
  {
    umem::Host host;
    assert_eq(host.is_ok(), true);
    umem::Address adr = host.alloc(11);
    assert_eq(host.is_ok(), true);
    adr.set(97, 5);
    assert_eq(host.is_ok(), true);
    (adr+5).set(98, 5);
    assert_eq(host.is_ok(), true);
    adr[10] = 0;
    assert_eq(host.is_ok(), true);
    assert_nstr_eq(10, (char*)adr, "aaaaabbbbb");
    assert_eq(host.is_ok(), true);
  }
  RETURN_STATUS;
}
