    //                  Entry point for the fully connected layer application

    //                  CR0 - base address of input activations = 0
    //                  Each input is 128 bytes (128 activations, 1 byte each) - we have 256 inputs
    //                  128 * 256 = 32768 = 0x8000
    //                  [input_idx][neuron] - for input index - jump in 128 bytes

    //                  Starts in next bank (128 * 1024 = 0x20000)
    //                  CR1 - base of weights - each weight is 128 bytes - 64 bytes overall
    //                  64 * 128 = 8192 = 0x2000

    //                  Starts in next bank (2 * 128 * 1024 = 0x40000)
    //                  CR2 - base of output activations - each output is 64 words
    //                  256 * 64 * 4 = 65536 = 0x10000
    //                  [output_idx][neuron][byte_idx_in_word]

    //                  Starts from reading first input
start:
    //                  Clear output activations to zero
    //                  TODO - add reset command for R registers

    //                  Input index counter
    set                 lr3 0 ;; // Input index counter = 0 - Base of inputs activations
    set                 lr1 0x8000 ;;
    set                 lr0 0x40000;;

input_loop:
    ldr                 r0 lr3 cr0;;

    set                 lr4 0x20000 ;; // Weight index counter = 0x20000 - Base of weights
    set                 lr5 0x21fff ;; // Weight limit
    set                 lr6 0 ;; // Output neuron index counter

weight_loop:
    ldr                 mem_bypass lr4 cr0;
    incr                lr4 128;
    incr                lr6 1;
    mac.agg             rq8 r0 mem_bypass lr6;
    blt                 lr5 lr4 +1;;

    set                 lr4 0x40000 ;;

    str                 r8 lr0 cr0;
    incr                lr0 128;;
    str                 r9 lr0 cr0;
    incr                lr0 128;
    blt_else_go_forward lr3 lr1 weight_loop;;

end:
    bkpt;;