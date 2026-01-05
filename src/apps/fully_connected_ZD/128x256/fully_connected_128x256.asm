    {%      set INPUT_SIZE = 128 %}
    {%      set NUM_SAMPLES = 1 %}
    {%      set NUM_OUTPUT_NEURONS = 256 %}
    {%      set NEURONS_PER_PASS = 128 %}
    {%      set WEIGHT_ROW_SIZE = 256 %}

    {%      set input_idx = 'lr0' %}
    {%      set input_limit = 'lr1' %}
    {%      set weight_byte_offset = 'lr2' %}
    {%      set element_idx = 'lr3' %}
    {%      set last_element_idx = 'lr4' %}
    {%      set output_offset = 'lr5' %}
    {%      set zero_offset = 'lr6' %}
    {%      set neuron_pass = 'lr7' %}
    {%      set first_pass_check = 'lr8' %}
    {%      set neuron_limit = 'lr9' %}

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
    set     {{ input_idx }} 0 ;; // Input index counter
    set     {{ input_limit }} {{ INPUT_SIZE * NUM_SAMPLES }} ;; // Total inputs ({{ INPUT_SIZE }} bytes × {{ NUM_SAMPLES }} sample = {{ INPUT_SIZE * NUM_SAMPLES }})
    set     {{ zero_offset }} 0 ;; // Offset for loading zeros
    set     {{ output_offset }} 0 ;; // Output offset
    set     {{ neuron_limit }} {{ NUM_OUTPUT_NEURONS }} ;; // Total output neurons (256)

input_loop:
    // Load input vector (128 bytes) - reused for both passes
    ldr     {{ input_vec }} {{ input_idx }} {{ input_base }};;

    // Process 256 neurons in 2 passes of 128 neurons each
    set     {{ neuron_pass }} 0;; // Pass 0: neurons 0-127, Pass 1: neurons 128-255

neuron_pass_loop:
    // Clear accumulator registers by loading zeros (4 R registers = 512 bytes for 128 neurons)
    ldr     {{ result_r0 }} {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 128;;
    ldr     {{ result_r1 }} {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 256;;
    ldr     r10 {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 384;;
    ldr     r11 {{ zero_offset }} {{ zeros_base }};;
    set     {{ zero_offset }} 0;; // Reset for next iteration

    // Set weight offset base depending on pass (0 for first pass, 128 for second pass)
    // neuron_pass is 0 or 128
    set     {{ first_pass_check }} 1;;
    set     {{ weight_byte_offset }} 0;;
    blt     {{ neuron_pass }} {{ first_pass_check }} skip_offset;;  // If pass==0, skip adding offset
    set     {{ weight_byte_offset }} {{ NEURONS_PER_PASS }};; // Set to 128 for second pass
skip_offset:

    // Initialize element loop
    set     {{ element_idx }} 0 ;; // Input element index (0 to 127)
    set     {{ last_element_idx }} {{ INPUT_SIZE - 1 }} ;; // Last input element index (127)

element_loop:
    // Load 128 bytes of weights starting at weight_offset_base (neurons 0-127 or 128-255)
    ldr     mem_bypass {{ weight_byte_offset }} {{ weight_base }};
    incr    {{ weight_byte_offset }} {{ WEIGHT_ROW_SIZE }};
    incr    {{ element_idx }} 1;
    // Multiply-accumulate: result += weight_row * input[element]
    mac.ev {{ result_quad }} mem_bypass {{ input_vec }} {{ element_idx }};
    blt     {{ element_idx }} {{ last_element_idx }} element_loop;;

    //      After all weights processed, store results
    //      Results are in rq8 (r8-r11) = 128 neurons × 4 bytes = 512 bytes
    str     {{ result_r0 }} {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} 128;;
    str     {{ result_r1 }} {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} 128;;
    str     r10 {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} 128;;
    str     r11 {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} 128;;

    //      Move to next neuron pass
    incr    {{ neuron_pass }} {{ NEURONS_PER_PASS }};;
    blt     {{ neuron_pass }} {{ neuron_limit }} neuron_pass_loop;;

    //      Move to next input (only 1 sample in this example)
    incr    {{ input_idx }} {{ INPUT_SIZE }};;
    blt     {{ input_idx }} {{ input_limit }} input_loop;;

end:
    bkpt;;
