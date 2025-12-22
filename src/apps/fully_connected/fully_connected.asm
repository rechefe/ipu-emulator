    {%      set INPUT_SIZE = 128 %}
    {%      set NUM_INPUTS = 256 %}
    {%      set NUM_OUTPUT_NEURONS = 64 %}
    {%      set WEIGHT_SIZE = 128 %}

    {%      set input_idx = 'lr0' %}
    {%      set input_limit = 'lr1' %}
    {%      set weight_byte_offset = 'lr2' %}
    {%      set neuron_idx = 'lr3' %}
    {%      set last_neuron_idx = 'lr4' %}
    {%      set output_offset = 'lr5' %}

    {%      set input_vec = 'r0' %}
    {%      set result_quad = 'rq8' %}
    {%      set result_r0 = 'r8' %}
    {%      set result_r1 = 'r9' %}

    {%      set input_base = 'cr0' %}
    {%      set weight_base = 'cr1' %}
    {%      set output_base = 'cr2' %}

start:
    //      Clear output activations to zero
    //      TODO - add reset command for R registers

    //      Initialize loop counters
    set     {{ input_idx }} 0 ;; // Input index counter
    set     {{ input_limit }} {{ INPUT_SIZE * 10 }} ;; // Total inputs limit ({{ INPUT_SIZE }} bytes * 10 = {{ INPUT_SIZE * 10 }})

input_loop:
    ldr     {{ input_vec }} {{ input_idx }} {{ input_base }};;

    set     {{ weight_byte_offset }} 0 ;; // Output neuron index counter (in bytes - jumps in {{ WEIGHT_SIZE }})
    set     {{ neuron_idx }} 0 ;; // (In neurons - jumps in 1)
    set     {{ last_neuron_idx }} {{ NUM_OUTPUT_NEURONS - 1 }} ;; // Last output neuron index

weight_loop:
    ldr     mem_bypass {{ weight_byte_offset }} {{ weight_base }};
    incr    {{ weight_byte_offset }} {{ WEIGHT_SIZE }};
    incr    {{ neuron_idx }} 1;
    mac.agg {{ result_quad }} {{ input_vec }} mem_bypass {{ neuron_idx }};
    blt     {{ neuron_idx }} {{ last_neuron_idx }} weight_loop;;

    //      After all weights processed, store results
    //      Results are already in {{ result_quad }} ({{ result_r0 }}-r11)
    str     {{ result_r0 }} {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} {{ WEIGHT_SIZE }};;
    str     {{ result_r1 }} {{ output_offset }} {{ output_base }};
    incr    {{ output_offset }} {{ WEIGHT_SIZE }};;

    //      Move to next input
    incr    {{ input_idx }} {{ INPUT_SIZE }};;
    blt     {{ input_idx }} {{ input_limit }} input_loop;;

end:
    bkpt;;