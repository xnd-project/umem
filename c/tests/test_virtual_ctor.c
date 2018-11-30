#include "umem_testing.h"

int main() {
  umemVirtual virt;
  umemVirtual_ctor(&virt);
  assert(umem_is_ok(&virt));
  umemVirtual_dtor(&virt);
  RETURN_STATUS;
}
