#include "umem_testing.h"

int main() {
  {
    static char fn[] = TMPDIR "test_file_init_cxx.txt";

    umem::Host host;
    assert_eq(host.is_ok(), true);

    static char text[] = "abcdefghij\0";

    {
      umem::File file(fn, "wb");
      assert_eq(file.is_ok(), true);
      umem::Address addr = file.alloc(0); // open the file for writing, 0 size is arbitrary
      assert_eq(file.is_ok(), true);
      addr.copy_from(text, strlen(text));
      assert_eq(file.is_ok(), true);
    }

    {
      umem::File file(fn, "rb");
      assert_eq(file.is_ok(), true);
      umem::Address addr = file.alloc(0); // open the file for reading
      assert_eq(file.is_ok(), true);
      umem::Address addr2 = host.alloc(strlen(text)+1);
      addr2[strlen(text)] = '\0';
      addr.copy_to(addr2, strlen(text));
      assert_eq(host.is_ok(), true);
      assert_eq(file.is_ok(), true);
      assert_nstr_eq(strlen(text), (char*)addr2, text);
    }

    {
      umem::File file("", "rb");
      assert_eq(file.is_ok(), false);
      assert_str_eq(file.get_message().c_str(), "invalid 0-length filename");
      file.clear_status();
      assert_eq(file.is_ok(), true);
    }
    
  }
  RETURN_STATUS;
}
