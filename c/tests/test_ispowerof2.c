#include "umem_testing.h"

int main() {
  assert_int_eq(umem_ispowerof2(0), 0);
  assert_int_eq(umem_ispowerof2(1), 1);
  assert_int_eq(umem_ispowerof2(2), 1);
  assert_int_eq(umem_ispowerof2(3), 0);
  assert_int_eq(umem_ispowerof2(4), 1);
  assert_int_eq(umem_ispowerof2(5), 0);
  assert_int_eq(umem_ispowerof2(6), 0);
  assert_int_eq(umem_ispowerof2(7), 0);
  assert_int_eq(umem_ispowerof2(8), 1);
  assert_int_eq(umem_ispowerof2(9), 0);
  RETURN_STATUS;
}
