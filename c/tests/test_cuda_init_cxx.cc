#include "umem_testing.h"

int main() {
  {
    umem::Cuda dev(0);
    umem::Host host;
    assert_is_ok(dev);
    static char text[] = "abcdefghij";
    umem::Address addr = dev.alloc(10);
    assert_is_ok(dev);
    addr.copy_from(text, 10);
    assert_is_ok(dev);
    assert_is_ok(host);
    umem::Address addr2 = host.alloc(10);
    assert_is_ok(host);
    addr.copy_to(addr2, 10);
    assert_is_ok(host);
    assert_is_ok(dev);
    assert_str_eq((char*)addr2, "abcdefghij");
  }
  RETURN_STATUS;
}
