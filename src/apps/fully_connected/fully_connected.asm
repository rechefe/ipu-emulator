    //      Entry point for the fully connected layer application

    //      CR0 - base address of input activations = 0
    //      Each input is 128 bytes (128 activations, 1 byte each) - we have 256 inputs
    //      128 * 256 = 32768 = 0x8000
    //      [input_idx][neuron] - for input index - jump in 128 bytes

    //      Starts in next bank (128 * 1024 = 0x20000)
    //      CR1 - base of weights - each weight is 128 bytes - 64 bytes overall
    //      64 * 128 = 8192 = 0x2000

    //      Starts in next bank (2 * 128 * 1024 = 0x40000)
    //      CR2 - base of output activations - each output is 64 words
    //      256 * 64 * 4 = 65536 = 0x10000
    //      [output_idx][neuron][byte_idx_in_word]

    //      Starts from reading first input
start:
    //      Clear output activations to zero
    //      TODO - add reset command for R registers

    //      Build 0x40000 (262144) using multiple 16-bit increments
    set     lr0 0 ;; // Input index counter
    set     lr1 0x2000 ;; // Total inputs limit (256 * 128 = 32768 bytes)

input_loop:
    ldr     r0 lr0 cr0;;

    set     lr2 0 ;; // Output neuron index counter (in bytes - jumps in 128)
    set     lr3 0 ;; // (In neurons - jumps in 1)
    set     lr4 63 ;; // Last output neuron index

weight_loop:
    ldr     mem_bypass lr2 cr1;
    incr    lr2 128;
    incr    lr3 1;
    mac.agg rq8 r0 mem_bypass lr3;
    blt     lr3 lr4 weight_loop;;

    //      After all weights processed, store results
    //      Results are already in RQ8 (r8-r11)
    str     r8 lr5 cr2;
    incr    lr5 128;;
    str     r9 lr5 cr2;
    incr    lr5 128;;

    //      Move to next input
    incr    lr0 128;;
    blt     lr0 lr1 input_loop;;

end:
    bkpt;;