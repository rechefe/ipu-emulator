OUTPUT_ARCH(riscv)
ENTRY(_start)

RAM_BASE = 0x80000000;

SECTIONS {
    . = RAM_BASE;

    .text : {
        *(.text._start)
        *(.text .text.*)
    }

    .rodata : {
        *(.rodata .rodata.*)
    }

    .data : {
        *(.data .data.*)
    }

    .bss : {
        *(.bss .bss.*)
    }
}
