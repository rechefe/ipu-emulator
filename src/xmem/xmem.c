#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "xmem.h"

xmem__obj_t *xmem__initialize_xmem()
{
    xmem__obj_t *xmem;
    xmem = malloc(sizeof(xmem__obj_t));
    memset(xmem->bytes, 0, XMEM__XMEM_SIZE_BYTES);
    return xmem;
}

void xmem__load_matrix_to(xmem__obj_t *xmem, uint8_t *matrix, int rows, int cols, int start_address)
{
    for (int i = 0; i < rows; i++)
    {
        int addr_offset = XMEM__ALIGN_ADDR(i * cols);
        uint8_t *row_ptr = matrix + (i * cols);
        xmem__load_array_to(xmem, row_ptr, cols, start_address + addr_offset);
    }
}

void xmem__load_array_to(xmem__obj_t *xmem, uint8_t *array, int count, int start_address)
{
    for (int i = 0; i < count; i++)
    {
        int addr = start_address + i;
        xmem->bytes[addr] = array[i];
    }
}

void xmem__write_address(xmem__obj_t *xmem, int address, const uint8_t *data, int size)
{
    assert(address >= 0 && (address + size) <= XMEM__XMEM_SIZE_BYTES);
    memcpy(&xmem->bytes[address], data, size);
}

void xmem__read_address(xmem__obj_t *xmem, int address, uint8_t *data, int size)
{
    assert(address >= 0 && (address + size) <= XMEM__XMEM_SIZE_BYTES);
    memcpy(data, &xmem->bytes[address], size);
}
