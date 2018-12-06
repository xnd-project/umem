#include <stdio.h>
#include <inttypes.h>

void binDump(char *desc, void *addr, int len)
{
  int i, k=8, n=8;
  unsigned char *pc = (unsigned char*)addr;
  uint64_t i64;
  // Output description if given.
  if (desc != NULL)
    printf ("%s:\n", desc);
  
  // Process every byte in the data.
  for (i = 0; i < len; i++) {
    // Multiple of k means new line (with line offset).

    if (i && ((i % n) == 0)) {
      i64 = ((uint64_t*)(pc + i - 8))[0];
      printf("  %08" PRIxPTR "", i64);
    }
    
    if ((i % k) == 0) {
      // Just don't print ASCII for the zeroth line.
      if (i != 0)
        printf("  \n");
      
      // Output the offset.
      printf("  %04x ", i);
    }
    
    // Now the hex code for the specific character.
    printf("%02x:", pc[i]);
    for(size_t j = 0; j < 7; ++j)
    {
      putchar('0' + (((pc[i]) >> (j)) & 1));
    }
    putchar(' ');             
  }

  if (len % n == 0) {
    i64 = ((uint64_t*)(pc + len - 8))[0];
    printf("  %08" PRIxPTR "", i64);
  }
  
  // Pad out last line if not exactly 16 characters.
  while ((i % k) != 0) {
    printf("   ");
    i++;
  }
  
  printf("  \n");
}

/*
  https://gist.github.com/domnikl/af00cc154e3da1c5d965/b26402c240567c0e788a2ed8013ff21a56fc2980
 */
void hexDump(char *desc, void *addr, int len)
{
    int i;
    unsigned char buff[17];
    unsigned char *pc = (unsigned char*)addr;

    // Output description if given.
    if (desc != NULL)
        printf ("%s:\n", desc);

    // Process every byte in the data.
    for (i = 0; i < len; i++) {
        // Multiple of 16 means new line (with line offset).

        if ((i % 16) == 0) {
            // Just don't print ASCII for the zeroth line.
            if (i != 0)
                printf("  %s\n", buff);

            // Output the offset.
            printf("  %04x ", i);
        }

        // Now the hex code for the specific character.
        printf(" %02x", pc[i]);

        // And store a printable ASCII character for later.
        if ((pc[i] < 0x20) || (pc[i] > 0x7e)) {
            buff[i % 16] = '.';
        } else {
            buff[i % 16] = pc[i];
        }

        buff[(i % 16) + 1] = '\0';
    }

    // Pad out last line if not exactly 16 characters.
    while ((i % 16) != 0) {
        printf("   ");
        i++;
    }

    // And print the final ASCII bit.
    printf("  %s\n", buff);
}
