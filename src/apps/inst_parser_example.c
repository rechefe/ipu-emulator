#include <stdio.h>
#include <stdlib.h>
#include "src/tools/ipu-as-py/inst_parser.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <instruction_file>\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file '%s'\n", filename);
        return 1;
    }

    // Get file size to determine number of instructions
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    size_t inst_size = sizeof(inst_parser__inst_t);
    size_t num_instructions = file_size / inst_size;

    printf("File: %s\n", filename);
    printf("File size: %ld bytes\n", file_size);
    printf("Instruction size: %zu bytes\n", inst_size);
    printf("Number of instructions: %zu\n\n", num_instructions);

    for (size_t i = 0; i < num_instructions; i++) {
        inst_parser__inst_t inst;
        inst_parser__read_inst_from_file(file, &inst);
        
        printf("Instruction %zu:\n", i);
        inst_parser__print_inst(&inst);
        printf("\n");
    }

    fclose(file);
    return 0;
}
