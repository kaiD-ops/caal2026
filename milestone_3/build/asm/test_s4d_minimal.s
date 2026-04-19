
build/exe/test_s4d_minimal.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70040117          	auipc	sp,0x70040
80000004:	01010113          	addi	sp,sp,16 # f0040010 <_stack_start>
80000008:	70040297          	auipc	t0,0x70040
8000000c:	ff828293          	addi	t0,t0,-8 # f0040000 <test_A_real>
80000010:	0002a007          	flw	ft0,0(t0)
80000014:	70040297          	auipc	t0,0x70040
80000018:	ff028293          	addi	t0,t0,-16 # f0040004 <test_A_imag>
8000001c:	0002a087          	flw	ft1,0(t0)
80000020:	20000553          	fsgnj.s	fa0,ft0,ft0
80000024:	2061                	c.jal	800000ac <exp_f>
80000026:	20a50253          	fsgnj.s	ft4,fa0,fa0
8000002a:	20108553          	fsgnj.s	fa0,ft1,ft1
8000002e:	28cd                	c.jal	80000120 <cos_f>
80000030:	20a502d3          	fsgnj.s	ft5,fa0,fa0
80000034:	20108553          	fsgnj.s	fa0,ft1,ft1
80000038:	20d9                	c.jal	800000fe <sin_f>
8000003a:	20a50353          	fsgnj.s	ft6,fa0,fa0
8000003e:	105273d3          	fmul.s	ft7,ft4,ft5
80000042:	10627e53          	fmul.s	ft8,ft4,ft6
80000046:	4281                	c.li	t0,0
80000048:	f0028053          	fmv.w.x	ft0,t0
8000004c:	f00280d3          	fmv.w.x	ft1,t0
80000050:	20000553          	fsgnj.s	fa0,ft0,ft0
80000054:	201085d3          	fsgnj.s	fa1,ft1,ft1
80000058:	20738653          	fsgnj.s	fa2,ft7,ft7
8000005c:	21ce06d3          	fsgnj.s	fa3,ft8,ft8
80000060:	28f9                	c.jal	8000013e <complex_mul>
80000062:	3f8002b7          	lui	t0,0x3f800
80000066:	f0028253          	fmv.w.x	ft4,t0
8000006a:	f00282d3          	fmv.w.x	ft5,t0
8000006e:	10527353          	fmul.s	ft6,ft4,ft5
80000072:	00657153          	fadd.s	ft2,fa0,ft6
80000076:	20b581d3          	fsgnj.s	ft3,fa1,fa1
8000007a:	f0028253          	fmv.w.x	ft4,t0
8000007e:	f00282d3          	fmv.w.x	ft5,t0
80000082:	20420553          	fsgnj.s	fa0,ft4,ft4
80000086:	205285d3          	fsgnj.s	fa1,ft5,ft5
8000008a:	20210653          	fsgnj.s	fa2,ft2,ft2
8000008e:	203186d3          	fsgnj.s	fa3,ft3,ft3
80000092:	2075                	c.jal	8000013e <complex_mul>
80000094:	70040297          	auipc	t0,0x70040
80000098:	f7428293          	addi	t0,t0,-140 # f0040008 <test_output>
8000009c:	00a2a027          	fsw	fa0,0(t0)
800000a0:	d05802b7          	lui	t0,0xd0580
800000a4:	4305                	c.li	t1,1
800000a6:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057fea0>
800000aa:	a001                	c.j	800000aa <_start+0xaa>

800000ac <exp_f>:
800000ac:	400002b7          	lui	t0,0x40000
800000b0:	f0028053          	fmv.w.x	ft0,t0
800000b4:	28050553          	fmin.s	fa0,fa0,ft0
800000b8:	c00002b7          	lui	t0,0xc0000
800000bc:	f0028053          	fmv.w.x	ft0,t0
800000c0:	28051553          	fmax.s	fa0,fa0,ft0
800000c4:	3f8002b7          	lui	t0,0x3f800
800000c8:	f0028053          	fmv.w.x	ft0,t0
800000cc:	00a07053          	fadd.s	ft0,ft0,fa0
800000d0:	10a570d3          	fmul.s	ft1,fa0,fa0
800000d4:	3f0002b7          	lui	t0,0x3f000
800000d8:	f0028153          	fmv.w.x	ft2,t0
800000dc:	1020f0d3          	fmul.s	ft1,ft1,ft2
800000e0:	00107053          	fadd.s	ft0,ft0,ft1
800000e4:	10a0f0d3          	fmul.s	ft1,ft1,fa0
800000e8:	3e2ab2b7          	lui	t0,0x3e2ab
800000ec:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
800000f0:	f0028153          	fmv.w.x	ft2,t0
800000f4:	1020f0d3          	fmul.s	ft1,ft1,ft2
800000f8:	00107553          	fadd.s	fa0,ft0,ft1
800000fc:	8082                	c.jr	ra

800000fe <sin_f>:
800000fe:	20a50053          	fsgnj.s	ft0,fa0,fa0
80000102:	10a570d3          	fmul.s	ft1,fa0,fa0
80000106:	10a0f153          	fmul.s	ft2,ft1,fa0
8000010a:	3e2ab2b7          	lui	t0,0x3e2ab
8000010e:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
80000112:	f00281d3          	fmv.w.x	ft3,t0
80000116:	10317153          	fmul.s	ft2,ft2,ft3
8000011a:	08207553          	fsub.s	fa0,ft0,ft2
8000011e:	8082                	c.jr	ra

80000120 <cos_f>:
80000120:	3f8002b7          	lui	t0,0x3f800
80000124:	f0028053          	fmv.w.x	ft0,t0
80000128:	10a570d3          	fmul.s	ft1,fa0,fa0
8000012c:	3f0002b7          	lui	t0,0x3f000
80000130:	f0028153          	fmv.w.x	ft2,t0
80000134:	1020f0d3          	fmul.s	ft1,ft1,ft2
80000138:	08107553          	fsub.s	fa0,ft0,ft1
8000013c:	8082                	c.jr	ra

8000013e <complex_mul>:
8000013e:	10c57053          	fmul.s	ft0,fa0,fa2
80000142:	10d5f0d3          	fmul.s	ft1,fa1,fa3
80000146:	08107153          	fsub.s	ft2,ft0,ft1
8000014a:	10d57053          	fmul.s	ft0,fa0,fa3
8000014e:	10c5f0d3          	fmul.s	ft1,fa1,fa2
80000152:	001071d3          	fadd.s	ft3,ft0,ft1
80000156:	20210553          	fsgnj.s	fa0,ft2,ft2
8000015a:	203185d3          	fsgnj.s	fa1,ft3,ft3
8000015e:	8082                	c.jr	ra
