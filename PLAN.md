Hi cursor,

We need to start with some refactor - changing lots of the instructions.
I will write here instructions we need to change, and new instructions we need to add :)


First we are adding another pipeline stage - after the accumulator, 
This stage will be called "Activation & Quantization" - aaq
This stage will have its own registers -
4 general purpose "aaq registers" - each composed out of 32 bits 

This was done :)
First - for the `acc` command - I want to add another variant which includes an LR argument `aaq_rf_idx` which chooses which aaq register to choose - it copmoses the same `acc` command, only its adding `aaq_regs[aaq_rf_idx]` to each of the 128 words accumulated :)

`aaq_rf_idx` must range in the range of 4 (the amount of general purpose registers)
Think of a meaningful name for the instruction and let's add it to the specification, implement it and test it
Until here - all done

New instruction - in the AAQ stage

This instruction is called `aaq` - does activation and quantization, format

`aaq ActivationType, QuantizationType, 