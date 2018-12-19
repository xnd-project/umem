#include "umem_testing.h"

int main() {
  {
    umem::Host host;
    size_t n = 64;
    umem::Address adr = host.calloc(n, 1);
    char * ptr = adr;
    double d = ((double*)ptr)[0] = 3.7;
    assert_eq(((double*)adr)[0], d);
    ptr += sizeof(double);
    float f = ((float*)ptr)[0] = 3.7f;
    assert_eq(((float*)(adr+sizeof(double)))[0], f);
    ptr += sizeof(float);
    int i = ((int*)ptr)[0] = 1234;
    assert_eq(((int*)(adr+sizeof(double)+sizeof(float)))[0], i);
    ptr += sizeof(int);
    (adr+sizeof(double)+sizeof(float)+sizeof(int))[0] = (char)137;
    assert_eq(ptr[0], (char)137);

    assert_eq(adr == adr, true);
    assert_eq(adr == (adr+0), true);
    assert_eq(adr == (uintptr_t)adr, true);
    assert_eq(adr == 0, false);
  }
  RETURN_STATUS;
}
