
build/exe/test_math_simple.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70040117          	auipc	sp,0x70040
80000004:	00010113          	addi	sp,sp,0 # f0040000 <_stack_start>
80000008:	f0000553          	fmv.w.x	fa0,zero
8000000c:	2039                	c.jal	8000001a <exp_f>
8000000e:	d05802b7          	lui	t0,0xd0580
80000012:	4305                	c.li	t1,1
80000014:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057ff0e>
80000018:	a001                	c.j	80000018 <_start+0x18>

8000001a <exp_f>:
8000001a:	400002b7          	lui	t0,0x40000
8000001e:	f0028053          	fmv.w.x	ft0,t0
80000022:	28050553          	fmin.s	fa0,fa0,ft0
80000026:	c00002b7          	lui	t0,0xc0000
8000002a:	f0028053          	fmv.w.x	ft0,t0
8000002e:	28051553          	fmax.s	fa0,fa0,ft0
80000032:	3f8002b7          	lui	t0,0x3f800
80000036:	f0028053          	fmv.w.x	ft0,t0
8000003a:	00a07053          	fadd.s	ft0,ft0,fa0
8000003e:	10a570d3          	fmul.s	ft1,fa0,fa0
80000042:	3f0002b7          	lui	t0,0x3f000
80000046:	f0028153          	fmv.w.x	ft2,t0
8000004a:	1020f0d3          	fmul.s	ft1,ft1,ft2
8000004e:	00107053          	fadd.s	ft0,ft0,ft1
80000052:	10a0f0d3          	fmul.s	ft1,ft1,fa0
80000056:	3e2ab2b7          	lui	t0,0x3e2ab
8000005a:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
8000005e:	f0028153          	fmv.w.x	ft2,t0
80000062:	1020f0d3          	fmul.s	ft1,ft1,ft2
80000066:	00107553          	fadd.s	fa0,ft0,ft1
8000006a:	8082                	c.jr	ra

8000006c <sin_f>:
8000006c:	20a50053          	fsgnj.s	ft0,fa0,fa0
80000070:	10a570d3          	fmul.s	ft1,fa0,fa0
80000074:	10a0f153          	fmul.s	ft2,ft1,fa0
80000078:	3e2ab2b7          	lui	t0,0x3e2ab
8000007c:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
80000080:	f00281d3          	fmv.w.x	ft3,t0
80000084:	10317153          	fmul.s	ft2,ft2,ft3
80000088:	08207553          	fsub.s	fa0,ft0,ft2
8000008c:	8082                	c.jr	ra

8000008e <cos_f>:
8000008e:	3f8002b7          	lui	t0,0x3f800
80000092:	f0028053          	fmv.w.x	ft0,t0
80000096:	10a570d3          	fmul.s	ft1,fa0,fa0
8000009a:	3f0002b7          	lui	t0,0x3f000
8000009e:	f0028153          	fmv.w.x	ft2,t0
800000a2:	1020f0d3          	fmul.s	ft1,ft1,ft2
800000a6:	08107553          	fsub.s	fa0,ft0,ft1
800000aa:	8082                	c.jr	ra

800000ac <complex_mul>:
800000ac:	10c57053          	fmul.s	ft0,fa0,fa2
800000b0:	10d5f0d3          	fmul.s	ft1,fa1,fa3
800000b4:	08107153          	fsub.s	ft2,ft0,ft1
800000b8:	10d57053          	fmul.s	ft0,fa0,fa3
800000bc:	10c5f0d3          	fmul.s	ft1,fa1,fa2
800000c0:	001071d3          	fadd.s	ft3,ft0,ft1
800000c4:	20210553          	fsgnj.s	fa0,ft2,ft2
800000c8:	203185d3          	fsgnj.s	fa1,ft3,ft3
800000cc:	8082                	c.jr	ra

800000ce <tanh_f>:
800000ce:	1141                	c.addi	sp,-16
800000d0:	c606                	c.swsp	ra,12(sp)
800000d2:	00a57553          	fadd.s	fa0,fa0,fa0
800000d6:	3791                	c.jal	8000001a <exp_f>
800000d8:	3f8002b7          	lui	t0,0x3f800
800000dc:	f0028053          	fmv.w.x	ft0,t0
800000e0:	080575d3          	fsub.s	fa1,fa0,ft0
800000e4:	00057653          	fadd.s	fa2,fa0,ft0
800000e8:	18c5f553          	fdiv.s	fa0,fa1,fa2
800000ec:	40b2                	c.lwsp	ra,12(sp)
800000ee:	0141                	c.addi	sp,16
800000f0:	8082                	c.jr	ra
