# IPU Debug

Our new feature would be a debug

I want to add another instruction inside the compound instruction structure...

Basically I want the ability to break (conditionally or unconditionally) on certain instructions,

The compound instruction must include `break.*` instruction which can run in parallel next to the other instructions
This instruction must look at the `regfile` snapshot when its conditioned


For start, let's have two instructions - 

`break` - breaks unconditionally
`break.ifeq lr immediate` - that breaks if `lr` register is equals to an immediate

I want the emulator to have the debug feature enabled - and if its enabled, break must stop the execution and open a prompt:

```
debug >>> 
```

And here I want basically the ability to read and write register of all types

LEVEL 0 - print these to the screen
LEVEL 1 - print out the disassembled instructions (its best to change the generated `inst_parser` module to allow also for disassembling instructions - not only parsing them)
LEVEL 2 - save all registers to a JSON to allow debug with out tools

I basically would like this debug module to receive the ipu handler - its best if this would be a CPP module which uses pre-existing libraries for JSON and CLI arguments - it must be easy and scalable to add new commands to the CLI and should be done with the least amount of effort

GOOD LUCK :)