OUTPUT_ARCH( "riscv" )
ENTRY(_start)

SECTIONS
{
  /* 1. Code Segment Section (Text) */
  . = 0x80000000;
  .text.init . : { *(.text.init) }
  .text      . : { *(.text) }
  _end = .;

  /* 2. MMIO Peripheral Console I/O Window Section */
  . = 0xd0580000;
  .data.io . : { *(.data.io) }

  /* 3. Mutable Data and Activation State Section */
  . = 0xf0040000;
  .data : ALIGN(8) { 
      *(.data) 
      *(.rodata*) 
  }
  
  /* Ensure .bss is cleanly aligned to an 8-byte boundary to match e32 alignment requirements */
  .bss : ALIGN(8) { 
      *(.bss) 
      *(COMMON)
  }

  /* FIX: Explicitly place the stack ceiling pointer AFTER .bss has concluded, 
     adding a safe 32 KB (0x8000) padding buffer that grows cleanly downwards */
  . = ALIGN(16);
  _stack_base = .;
  . = . + 0x8000;
  STACK = .;

  /* 4. Peripheral Boot Controller/Vector ROM Hook Mapping */
  . = 0xfffffff8; 
  .data.ctl : { 
      LONG(0xf0040000); 
      LONG(STACK); 
  }
}