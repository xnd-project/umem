#include "umem_testing.h"

using namespace umem;
int main() {
  {
    Host host;
    Address adr = host.alloc(11);
    assert_eq(host.is_ok(), true);
    for (int i=0; i<10; ++i) adr[i] = i+97;
    adr[10] = 0;
    assert_str_eq((char*)adr, "abcdefghij");
    assert_eq(host.is_ok(), true);
  }
  RETURN_STATUS;
}
