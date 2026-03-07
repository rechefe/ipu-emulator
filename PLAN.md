I want to implement a new instruction (new slot in the compound instruction) under the new stage - Activation and Quantization

The first instruction I want to implement is called 

`aaq.agg`

Its role is to take the 128 words and collapse them into one word, and store it into one of the AAQ registers 

Fields - 

- Enum which says - SUM/MAX
- Post function - selects between the following (VALUE*CR, 1/VALUE,1/SQRT(VALUE),VALUE)
- CR_IDX - selects the CR in the post function (relevant only if this is the post function)
- AaqRegField - selects which Aaq register we are using 

The flow - 

If its SUM - take the sum of all the values in RACC, if its max, take the MAX value out of all of them,

In the MAX - we need to also include the old value of the AAQ_REG we want to store to in the max check (if its already the maximum - no reason to update it)

perform the post function on the result of the MAX/SUM
Store in the AaqReg