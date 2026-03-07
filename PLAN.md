I want a new instruction called `acc.stride` - 

This instructions role is to **reorder** the multiplication result as it arrives to the accumulator stage into `racc`. Its role is to take only partial data from the product result, and store it based on some pattern delivered by the fields of the instructions inside RACC

We need only to update specific RACC indexes - and leave unchanged the rest.

Let's get into the details of how the stride works, we have the following arguments to the instructions

- First we have ELEMENTS_IN_ROW - its an enum that should choose between 8, 16, 32, 64
- Than we have two different enums, one for HORIZONTAL stride, and one for VERTICAL stride, both selecting bools - (1) is this type of stride enabled, (2) and is it inverted (only relevant if enabled), (3) - relevant only for HORIZONTAL, chooses if we want the stride expands or doesn't expand.
- Last, we have the offset - comes from LR - we take % 4 of it, if its 0 we store the values starting from index 0 in RACC, if its 1 we store from index 32, 2 -> index 64, and 3 -> index 96

It goes the following way:

The purpose of the stride is to decimate the inputs, we treat the input as an image with ROWs and COLUMNs, we want to be able to take every 2nd row and every 2nd column and ignore the rest

We start with horizontal stride - 
Say the multiplication result is composed out of 
W0 W1 W2 W3 .... W127 - each is 32 bits

And `ELEMENTS_IN_ROW` is 8.
Than we want to drop every second item in every row.

W0 W2 W4 ... W126 -> Resulting in 64 elements

This is if we are inverted, if we are its the other way around

W1 W3 W5 ... W127 -> Resulting in 64 elements

Now there's the expand, we'll sometimes want to keep the size of a row - so we'll pad with 0s, in case we're not inverted and we have 8 elements in a row - it will look like that:

W0 W2 W4 W6 0 0 0 0 W8 W10 W12 W14 0 0 0 0 ... -> Resulting in 128 elements

Next we move to vertical stride -> Here we wont to decimate the rows, we want to take every 2nd row.

So say we did not have horizontal stride - and ELEMENTS_IN_ROW is 8 - we'll have the following outcome for not inverted 

W0 W1 W2 .. W7 W16 W17 W18 .. W23 ... 

In case it is inverted we'll have - 

W8 W9 W10 ... W15 W24 W25 W26 ... W31 ...

In both cases, if horizontal stride is enabled - it will results in half the items. 
This is done on the result of the horizontal stride, say we did not expand - than this stage needs to remember that a row in the context of it got divided by 2, and say the horizntal stride had expansion - the row size is the same... 

The output of this stage is on of the following number elements - 32 / 64 / 128 

32 - if both are enabled (and horizontal does not have expansion)
64 - if only one is enabled (horizontal - only if it does not have expansion)
128 - if none is enabled, or horizontal has expanision and vertical disabled

The result indexes will be stored into RACC starting from the offset arriving from the last argument
We only store the amount of indexes needed, we do not change indexes exceeding our result size and starting index (both offset and size need to be taken into consideration)