#ifndef XMEM_H
#define XMEM_H

#include <stdint.h>
#include "assert.h"

#define XMEM__XMEM_SIZE_BYTES (1 << 21) // 2 MB
#define XMEM__XMEM_WIDTH_BYTES 128
#define XMEM__XMEM_DEPTH_WORDS ((XMEM__XMEM_SIZE_BYTES / XMEM__XMEM_WIDTH_BYTES))

#define XMEM__ALIGN_ADDR(addr) \
    (((addr) % XMEM__XMEM_WIDTH_BYTES == 0) ? (addr) : ((addr) + XMEM__XMEM_WIDTH_BYTES - ((addr) % XMEM__XMEM_WIDTH_BYTES)))
#define XMEM__WORDS_NEEDED_FOR_BYTES(bytes) \
    XMEM__ALIGN_ADDR(bytes) / XMEM__XMEM_WIDTH_BYTES

typedef union
{
    uint8_t bytes[XMEM__XMEM_SIZE_BYTES];
    uint8_t words[XMEM__XMEM_DEPTH_WORDS][XMEM__XMEM_WIDTH_BYTES];
} xmem__obj_t;

xmem__obj_t *xmem__initialize_xmem();
void xmem__load_matrix_to(xmem__obj_t *xmem, uint8_t *matrix, int rows, int cols, int start_address);
void xmem__load_array_to(xmem__obj_t *xmem, uint8_t *array, int count, int start_address);

void xmem__write_address(xmem__obj_t *xmem, int address, const uint8_t *data, int size);
void xmem__read_address(xmem__obj_t *xmem, int address, uint8_t *data, int size);

#endif // XMEM_H
