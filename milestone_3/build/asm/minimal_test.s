
build/exe/minimal_test.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	4281                	c.li	t0,0
80000002:	4329                	c.li	t1,10

80000004 <test_loop>:
80000004:	0285                	c.addi	t0,1
80000006:	fe62cfe3          	blt	t0,t1,80000004 <test_loop>
8000000a:	4501                	c.li	a0,0
8000000c:	48a9                	c.li	a7,10
8000000e:	00000073          	ecall
