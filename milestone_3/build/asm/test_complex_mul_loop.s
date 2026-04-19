
build/exe/test_complex_mul_loop.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70040117          	auipc	sp,0x70040
80000004:	00010113          	addi	sp,sp,0 # f0040000 <_stack_start>
80000008:	3f8002b7          	lui	t0,0x3f800
8000000c:	f0028053          	fmv.w.x	ft0,t0
80000010:	f00280d3          	fmv.w.x	ft1,t0
80000014:	f0028153          	fmv.w.x	ft2,t0
80000018:	f00281d3          	fmv.w.x	ft3,t0
8000001c:	4401                	c.li	s0,0

8000001e <loop>:
8000001e:	42a9                	c.li	t0,10
80000020:	00545d63          	bge	s0,t0,8000003a <done>
80000024:	20000553          	fsgnj.s	fa0,ft0,ft0
80000028:	201085d3          	fsgnj.s	fa1,ft1,ft1
8000002c:	20210653          	fsgnj.s	fa2,ft2,ft2
80000030:	203186d3          	fsgnj.s	fa3,ft3,ft3
80000034:	2809                	c.jal	80000046 <complex_mul>
80000036:	0405                	c.addi	s0,1
80000038:	b7dd                	c.j	8000001e <loop>

8000003a <done>:
8000003a:	d05802b7          	lui	t0,0xd0580
8000003e:	4305                	c.li	t1,1
80000040:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057ff98>
80000044:	a001                	c.j	80000044 <done+0xa>

80000046 <complex_mul>:
80000046:	10c57053          	fmul.s	ft0,fa0,fa2
8000004a:	10d5f0d3          	fmul.s	ft1,fa1,fa3
8000004e:	08107153          	fsub.s	ft2,ft0,ft1
80000052:	10d57053          	fmul.s	ft0,fa0,fa3
80000056:	10c5f0d3          	fmul.s	ft1,fa1,fa2
8000005a:	001071d3          	fadd.s	ft3,ft0,ft1
8000005e:	20210553          	fsgnj.s	fa0,ft2,ft2
80000062:	203185d3          	fsgnj.s	fa1,ft3,ft3
80000066:	8082                	c.jr	ra
