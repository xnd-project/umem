#include "umem_testing.h"

int main() {
  {
    umem::Host host;
    assert_eq(host.is_ok(), true);
    {
      umem::Address addr1 = host.alloc(11);
      umem::Address addr2 = host.alloc(11);
      assert_eq(host.is_ok(), true);
      for (char i=0; i<10; ++i) ((char*)addr1)[(int)i] = i+97;
      ((char*)addr1)[10] = 0;
      assert_eq(host.is_ok(), true);
      assert_str_eq((char*)addr1, "abcdefghij");
      assert_eq(host.is_ok(), true);
      addr1.copy_to(addr2, 11);
      assert_eq(host.is_ok(), true);
      assert_str_eq((char*)addr2, "abcdefghij");
      assert_str_eq((char*)addr2, (char*)addr1);
    }
    assert_eq(host.is_ok(), true);
  }
  RETURN_STATUS;
}
