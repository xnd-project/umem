#include "umem_testing.h"

int main() {
  {
    umem::Host host;
    assert_is_ok(host);
    {
      umem::Address addr1 = host.alloc(11);
      umem::Address addr2 = host.alloc(11);
      assert_is_ok(host);
      for (char i=0; i<10; ++i) ((char*)addr1)[(int)i] = i+97;
      ((char*)addr1)[10] = 0;
      assert_is_ok(host);
      assert_str_eq((char*)addr1, "abcdefghij");
      assert_is_ok(host);
      addr1.copy_to(addr2, 11);
      assert_is_ok(host);
      assert_str_eq((char*)addr2, "abcdefghij");
      assert_str_eq((char*)addr2, (char*)addr1);
    }
    assert_is_ok(host);
  }
  RETURN_STATUS;
}
