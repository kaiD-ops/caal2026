/* ============================================================================
 * linker.ld - RISC-V Linker Script for VeeR-iSS
 * ============================================================================
 *
 * This linker script defines the memory layout and sections for RISC-V 
 * programs running on the VeeR-iSS simulator.
 *
 * Typical VeeR-iSS memory layout:
 *   0x00000000 - 0x00400000  (4MB)    - Code/Text (ROM or cacheable)
 *   0x80000000 - 0x80800000  (8MB)    - SRAM (Data, heap, stack)
 *
 * Section mapping:
 *   .text   -> Code section (ROM)
 *   .data   -> Initialized data (SRAM)
 *   .bss    -> Uninitialized data (SRAM)
 */

/* ============================================================================
 * Memory Definition
 * ============================================================================ */

MEMORY {
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 32M
    RAM (rwx)   : ORIGIN = 0x80000000, LENGTH = 32M
}

/* ============================================================================
 * Entry Point
 * ============================================================================ */

ENTRY(main)

/* ============================================================================
 * Sections
 * ============================================================================ */

SECTIONS {
    /* Code and Read-Only Data */
    .text : {
        *(.text*)
        *(.rodata*)
    } >FLASH

    /* Initialized Data */
    .data : {
        *(.data*)
    } >RAM AT >FLASH
    
    /* Uninitialized Data and BSS */
    .bss : {
        *(.bss*)
        *(COMMON)
    } >RAM

    /* Debug Information (if present) */
    .debug_info  0 : { *(.debug_info)  }
    .debug_abbrev 0 : { *(.debug_abbrev) }
    .debug_line  0 : { *(.debug_line)  }
    .debug_frame 0 : { *(.debug_frame) }
}

/* ============================================================================
 * Symbol Definitions
 * ============================================================================ */

/* End of text section (first address past code) */
_etext = .;

/* Beginning and end of data section (in SRAM) */
_sdata = ADDR(.data);
_edata = ADDR(.data) + SIZEOF(.data);

/* Beginning and end of BSS section */
_sbss = ADDR(.bss);
_ebss = ADDR(.bss) + SIZEOF(.bss);

/* Stack and Heap */
_heap_start = _ebss;
_heap_end   = 0x80800000 - 1024;  /* Leave 1KB for stack */
_stack_end  = 0x80800000;
