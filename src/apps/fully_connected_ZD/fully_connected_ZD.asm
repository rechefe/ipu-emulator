    {%      set INPUT_SIZE = 128 %}
    {%      set NUM_INPUTS = 256 %}
    {%      set NUM_OUTPUT_NEURONS = 128 %}
    {%      set WEIGHT_SIZE = 128 %}

    {%      set input_idx = 'lr0' %}
    {%      set input_limit = 'lr1' %}
    {%      set weight_byte_offset = 'lr2' %}
    {%      set element_idx = 'lr3' %}
    {%      set last_element_idx = 'lr4' %}
    {%      set output_offset = 'lr5' %}
    {%      set zero_offset = 'lr6' %}

    {%      set input_vec = 'r0' %}
    {%      set result_quad = 'rq8' %}
    {%      set result_r0 = 'r8' %}
    {%      set result_r1 = 'r9' %}

    {%      set input_base = 'cr0' %}
    {%      set weight_base = 'cr1' %}
    {%      set output_base = 'cr2' %}
    {%      set zeros_base = 'cr3' %}

start:
    //      Clear output activations to zero
    //      TODO - add reset command for R registers

    //      Initialize loop counters
    set     {{ input_idx }} 0 ;; // Input index counter
    set     {{ input_limit }} {{ INPUT_SIZE * 10 }} ;; // Total inputs limit ({{ INPUT_SIZE }} bytes * 10 = {{ INPUT_SIZE * 10 }})
    set     {{ zero_offset }} 0 ;; // Offset for loading zeros

input_loop:
    // Clear accumulator registers by loading zeros (4 R registers = 512 bytes)
    ldr     {{ result_r0 }} {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 128;;
    ldr     {{ result_r1 }} {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 256;;
    ldr     r10 {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 384;;
    ldr     r11 {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 0;; // Reset for next iteration

    ldr     {{ input_vec }} {{ input_idx }} {{ input_base }};;

    set     {{ weight_byte_offset }} 0 ;; // Weight row offset (in bytes - jumps in {{ WEIGHT_SIZE }})
    set     {{ element_idx }} 0 ;; // Input element index (0 to 127)
    set     {{ last_element_idx }} {{ INPUT_SIZE - 1 }} ;; // Last input element index (127)

element_loop:
    ldr     mem_bypass {{ weight_byte_offset }} {{ weight_base }};
    incr    {{ weight_byte_offset }} {{ WEIGHT_SIZE }};
    incr    {{ element_idx }} 1;
    mac.ev {{ result_quad }} mem_bypass {{ input_vec }} {{ element_idx }};
    blt     {{ element_idx }} {{ last_element_idx }} element_loop;;

    //      After all weights processed, store results
    //      Results are in {{ result_quad }} (r8-r11) = 128 neurons × 4 bytes = 512 bytes
    str     {{ result_r0 }} {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} {{ WEIGHT_SIZE }};;
    str     {{ result_r1 }} {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} {{ WEIGHT_SIZE }};;
    str     r10 {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} {{ WEIGHT_SIZE }};;
    str     r11 {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} {{ WEIGHT_SIZE }};;

    //      Move to next input
    incr    {{ input_idx }} {{ INPUT_SIZE }};;
    blt     {{ input_idx }} {{ input_limit }} input_loop;;

end:
    bkpt;;