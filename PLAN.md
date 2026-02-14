# Refactoring 

I want to refactor the tools currently sitting in two different python packages - 

- `ipu-as-py`
- `ipu-emu-py`

Into one package

## Concerns
My main concerns is the duplication of content, I want instruction structure declaration and implementation (how does it executes) to sit in the same place... 

I want register definitions as `enum-token`-s inside the assembler to correlate to the registers in the context of the IPU.

I would like to plan a layout for the new joint python package - and steps for implementing it.