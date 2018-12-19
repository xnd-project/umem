#include "umem_testing.h"

int main() {
  {
    umem::Host host, host2;
    assert_is_ok(host);
    assert_is_ok(host2); 
    assert_eq(host == host, true);
    assert_eq(host == host2, true);
    assert_eq(host != host2, false);

    std::string fn1 = TMPDIR "test_is_same_context_1.txt";
    std::string fn2 = TMPDIR "test_is_same_context_2.txt";
    std::string fn21 = TMPDIR "test_is_same_context_2.txt";
    umem::File file1(fn1, "wb"), file2(fn2, "wb"), file21(fn21, "wb");
    assert_is_ok(file1);
    assert_is_ok(file2);
    assert_is_ok(file21);
    
    assert_eq(host == file1, false);
    assert_eq(file1 == host, false);
    assert_eq(file1 == file1, true);
    assert_eq(file1 == file2, false);
    assert_eq(file2 == file21, true);
    
    assert_is_ok(host);
    assert_is_ok(host2);
    assert_is_ok(file1);
    assert_is_ok(file2);
    assert_is_ok(file21);
  }
  RETURN_STATUS;
}
