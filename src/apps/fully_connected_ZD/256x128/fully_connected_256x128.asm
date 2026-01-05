    {%      set INPUT_SIZE = 256 %}
    {%      set INPUT_CHUNK_SIZE = 128 %}
    {%      set NUM_SAMPLES = 1 %}
    {%      set NUM_OUTPUT_NEURONS = 128 %}
    {%      set WEIGHT_ROW_SIZE = 128 %}

    {%      set input_idx = 'lr0' %}
    {%      set input_limit = 'lr1' %}
    {%      set weight_byte_offset = 'lr2' %}
    {%      set element_idx = 'lr3' %}
    {%      set last_element_idx = 'lr4' %}
    {%      set output_offset = 'lr5' %}
    {%      set zero_offset = 'lr6' %}
    {%      set input_chunk = 'lr7' %}
    {%      set input_chunk_limit = 'lr8' %}
    {%      set input_offset = 'lr9' %}

    {%      set input_vec = 'r0' %}
    {%      set result_quad = 'rq8' %}
    {%      set result_r0 = 'r8' %}
    {%      set result_r1 = 'r9' %}

    {%      set input_base = 'cr0' %}
    {%      set weight_base = 'cr1' %}
    {%      set output_base = 'cr2' %}
    {%      set zeros_base = 'cr3' %}

start:
    //      Initialize loop counters
    set     {{ input_idx }} 0 ;; // Input sample index counter
    set     {{ input_limit }} {{ INPUT_SIZE * NUM_SAMPLES }} ;; // Total inputs ({{ INPUT_SIZE }} bytes × {{ NUM_SAMPLES }} sample = {{ INPUT_SIZE * NUM_SAMPLES }})
    set     {{ zero_offset }} 0 ;; // Offset for loading zeros
    set     {{ output_offset }} 0 ;; // Output offset
    set     {{ input_chunk_limit }} 2 ;; // Process 2 chunks of 128 input elements

input_loop:
    // Clear accumulator registers by loading zeros (4 R registers = 512 bytes for 128 neurons)
    // We only clear once at the start, then accumulate across both input chunks
    ldr     {{ result_r0 }} {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 128;;
    ldr     {{ result_r1 }} {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 256;;
    ldr     r10 {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 384;;
    ldr     r11 {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 0;; // Reset for next iteration

    // Process 256 input elements in 2 chunks of 128 elements each
    set     {{ input_chunk }} 0;; // Chunk 0: elements 0-127, Chunk 1: elements 128-255
    set     {{ input_offset }} 0;; // Start at offset 0 for first chunk
    set     {{ weight_byte_offset }} 0 ;; // Weight row offset (continues across chunks)

input_chunk_loop:
    // Load input vector chunk (128 bytes) at current offset from input base
    ldr     {{ input_vec }} {{ input_offset }} {{ input_base }};;

    // Initialize element loop for this chunk
    set     {{ element_idx }} 0 ;; // Input element index within chunk (0 to 127)
    set     {{ last_element_idx }} {{ INPUT_CHUNK_SIZE - 1 }} ;; // Last input element index in chunk (127)

element_loop:
    // Load 128 bytes of weights (all neuron weights for this input element)
    // weight_byte_offset continues from 0 to (256 * 128) bytes across both chunks
    ldr     mem_bypass {{ weight_byte_offset }} {{ weight_base }};
    incr    {{ weight_byte_offset }} {{ WEIGHT_ROW_SIZE }};
    incr    {{ element_idx }} 1;
    // Multiply-accumulate: result += weight_row * input[element]
    // This accumulates across both input chunks
    mac.ev {{ result_quad }} mem_bypass {{ input_vec }} {{ element_idx }};
    blt     {{ element_idx }} {{ last_element_idx }} element_loop;;

    //      Move to next input chunk
    incr    {{ input_chunk }} 1;;
    incr    {{ input_offset }} {{ INPUT_CHUNK_SIZE }};; // Move to next 128 elements (offset += 128)
    blt     {{ input_chunk }} {{ input_chunk_limit }} input_chunk_loop;;

    //      After all input chunks processed, store accumulated results
    //      Results are in rq8 (r8-r11) = 128 neurons × 4 bytes = 512 bytes
    str     {{ result_r0 }} {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} {{ WEIGHT_ROW_SIZE }};;
    str     {{ result_r1 }} {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} {{ WEIGHT_ROW_SIZE }};;
    str     r10 {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} {{ WEIGHT_ROW_SIZE }};;
    str     r11 {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} {{ WEIGHT_ROW_SIZE }};;

    //      Move to next input sample (only 1 sample in this example)
    incr    {{ input_idx }} {{ INPUT_SIZE }};;
    blt     {{ input_idx }} {{ input_limit }} input_loop;;

end:
    bkpt;;
