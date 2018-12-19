#include "umem_testing.h"

int main() {
  {
    umem::Cuda dev(0);
    umem::Host host;
    assert_eq(dev.is_ok(), true);
    static char text[] = "abcdefghij";
    umem::Address addr = dev.alloc(10);
    assert_eq(dev.is_ok(), true);
    addr.copy_from(text, 10);
    assert_eq(dev.is_ok(), true);
    assert_eq(host.is_ok(), true);
    umem::Address addr2 = host.alloc(10);
    assert_eq(host.is_ok(), true);
    addr.copy_to(addr2, 10);
    assert_eq(host.is_ok(), true);
    assert_eq(dev.is_ok(), true);
    assert_str_eq((char*)addr2, "abcdefghij");
  }
  RETURN_STATUS;
}
