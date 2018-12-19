#include "umem_testing.h"

int main() {
  {
    static char fn[] = TMPDIR "test_file_init_cxx.txt";

    umem::Host host;
    assert_is_ok(host);

    static char text[] = "abcdefghij\0";

    {
      umem::File file(fn, "wb");
      assert_is_ok(file);
      umem::Address addr = file.alloc(10); // open the file for writing, 10 size is arbitrary
      assert_is_ok(file);
      addr.copy_from(text, strlen(text));
      assert_is_ok(file);
    }

    {
      umem::File file(fn, "rb");
      assert_is_ok(file);
      umem::Address addr = file.alloc(10); // open the file for reading
      assert_is_ok(file);
      umem::Address addr2 = host.alloc(strlen(text)+1);
      addr2[strlen(text)] = '\0';
      addr.copy_to(addr2, strlen(text));
      assert_is_ok(host);
      assert_is_ok(file);
      assert_nstr_eq(strlen(text), (char*)addr2, text);
    }

    {
      umem::File file("", "rb");
      assert_is_not_ok(file);
      assert_str_eq(file.get_message().c_str(), "invalid 0-length filename");
      file.clear_status();
      assert_is_ok(file);
    }
    
  }
  RETURN_STATUS;
}
