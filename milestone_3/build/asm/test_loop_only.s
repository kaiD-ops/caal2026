
build/exe/test_loop_only.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70040117          	auipc	sp,0x70040
80000004:	00010113          	addi	sp,sp,0 # f0040000 <_stack_start>
80000008:	4401                	c.li	s0,0

8000000a <outer>:
8000000a:	4289                	c.li	t0,2
8000000c:	02545263          	bge	s0,t0,80000030 <outer_done>
80000010:	4481                	c.li	s1,0

80000012 <inner>:
80000012:	4291                	c.li	t0,4
80000014:	0054dc63          	bge	s1,t0,8000002c <inner_done>
80000018:	3f8002b7          	lui	t0,0x3f800
8000001c:	f0028053          	fmv.w.x	ft0,t0
80000020:	f00280d3          	fmv.w.x	ft1,t0
80000024:	00107153          	fadd.s	ft2,ft0,ft1
80000028:	0485                	c.addi	s1,1
8000002a:	b7e5                	c.j	80000012 <inner>

8000002c <inner_done>:
8000002c:	0405                	c.addi	s0,1
8000002e:	bff1                	c.j	8000000a <outer>

80000030 <outer_done>:
80000030:	d05802b7          	lui	t0,0xd0580
80000034:	4305                	c.li	t1,1
80000036:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057ffc4>
8000003a:	a001                	c.j	8000003a <outer_done+0xa>
