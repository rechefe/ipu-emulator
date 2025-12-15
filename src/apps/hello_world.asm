# IPU Hello World - Simple Example
# This program demonstrates:
# 1. Setting up an address register
# 2. Storing register R4 (pre-initialized in C) to memory at 0x2000
# 3. Halting
#
# Note: R4 is already initialized with pattern 0,1,2,3...15,0,1,2...
# by the setup function in the C code

# Step 1: Set LR1 to point to output address (0x2000)
start:
    set  lr1 0x2000 ;;

# Step 2: Store register R4 to memory address 0x2000
# The MAC instruction provides the R register (r4) for the STR
# After this, memory at 0x2000 will contain R4's data
store:
    ldr  r0 lr1 cr0; mac.ee rq8 r4 r4 ;;

# Step 3: Stop execution
halt:
    bkpt
