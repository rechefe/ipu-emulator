    //  Entry point for the fully connected layer application

    //  CR0 - base address of input activations = 0
    //  Each input is 128 bytes (128 activations, 1 byte each) - we have 256 inputs
    //  128 * 256 = 32768 = 0x8000
    //  [input_idx][neuron] - for input index - jump in 128 bytes

    //  Starts in next bank (128 * 1024 = 0x20000)
    //  CR1 - base of weights - each weight is 128 bytes - 64 bytes overall
    //  64 * 128 = 8192 = 0x2000

    //  Starts in next bank (2 * 128 * 1024 = 0x40000)
    //  CR2 - base of output activations - each output is 64 words
    //  256 * 64 * 4 = 65536 = 0x10000
    //  [output_idx][neuron][byte_idx_in_word]

    //  Starts from reading first input
start:
    set lr0 0x0 ;; // Input base address
    set lr1 0x20000 ;; // Weights base address
    set lr2 0x40000 ;; // Output base address

    // Clear output activations to zero
    // TODO - add reset command for R registers


    // Input index counter
    set lr3 0 ;; // input_idx = 0

input_loop:
    ldr lr3 cr0;;