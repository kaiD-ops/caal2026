OUTPUT_ARCH( "riscv" )
ENTRY(_start)
SECTIONS
{
  . = 0x80000000;
  .text.init .  : { *(.text.init) }

  .text . : { *(.text) }
  _end = .;
  . = 0xd0580000;
  .data.io .  : { *(.data.io) }
    . = 0xf0040000 ;
  .data  :  ALIGN(0x800) { *(.*data) *(.rodata*) STACK = ALIGN(16) + 0x8000; }
  __bss_start = .;
  .bss : { *(.bss) }
  __bss_end = .;

    . = 0xfffffff8; .data.ctl : { LONG(0xf0040000); LONG(STACK) }
}
