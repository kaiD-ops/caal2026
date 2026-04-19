
build/exe/test_d_model_64.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70041117          	auipc	sp,0x70041
80000004:	a0010113          	addi	sp,sp,-1536 # f0040a00 <_stack_start>
80000008:	70040417          	auipc	s0,0x70040
8000000c:	7f840413          	addi	s0,s0,2040 # f0040800 <h_real>
80000010:	70041497          	auipc	s1,0x70041
80000014:	8f048493          	addi	s1,s1,-1808 # f0040900 <h_imag>
80000018:	4281                	c.li	t0,0

8000001a <init_loop>:
8000001a:	04000313          	addi	t1,zero,64
8000001e:	0062de63          	bge	t0,t1,8000003a <init_done>
80000022:	00229393          	slli	t2,t0,0x2
80000026:	00740e33          	add	t3,s0,t2
8000002a:	000e2023          	sw	zero,0(t3)
8000002e:	00748e33          	add	t3,s1,t2
80000032:	000e2023          	sw	zero,0(t3)
80000036:	0285                	c.addi	t0,1
80000038:	b7cd                	c.j	8000001a <init_loop>

8000003a <init_done>:
8000003a:	4901                	c.li	s2,0
8000003c:	4989                	c.li	s3,2

8000003e <t_loop>:
8000003e:	11395663          	bge	s2,s3,8000014a <done>
80000042:	4a01                	c.li	s4,0
80000044:	04000a93          	addi	s5,zero,64

80000048 <d_loop>:
80000048:	0f5a5f63          	bge	s4,s5,80000146 <d_done>
8000004c:	70040297          	auipc	t0,0x70040
80000050:	1b428293          	addi	t0,t0,436 # f0040200 <test_A_real>
80000054:	002a1313          	slli	t1,s4,0x2
80000058:	006283b3          	add	t2,t0,t1
8000005c:	0003a007          	flw	ft0,0(t2)
80000060:	70040297          	auipc	t0,0x70040
80000064:	2a028293          	addi	t0,t0,672 # f0040300 <test_A_imag>
80000068:	006283b3          	add	t2,t0,t1
8000006c:	0003a087          	flw	ft1,0(t2)
80000070:	20000553          	fsgnj.s	fa0,ft0,ft0
80000074:	20cd                	c.jal	80000156 <exp_f>
80000076:	20a50253          	fsgnj.s	ft4,fa0,fa0
8000007a:	20108553          	fsgnj.s	fa0,ft1,ft1
8000007e:	22b1                	c.jal	800001ca <cos_f>
80000080:	20a502d3          	fsgnj.s	ft5,fa0,fa0
80000084:	20108553          	fsgnj.s	fa0,ft1,ft1
80000088:	2205                	c.jal	800001a8 <sin_f>
8000008a:	20a50353          	fsgnj.s	ft6,fa0,fa0
8000008e:	105273d3          	fmul.s	ft7,ft4,ft5
80000092:	10627e53          	fmul.s	ft8,ft4,ft6
80000096:	006403b3          	add	t2,s0,t1
8000009a:	0003a007          	flw	ft0,0(t2)
8000009e:	006483b3          	add	t2,s1,t1
800000a2:	0003a087          	flw	ft1,0(t2)
800000a6:	20000553          	fsgnj.s	fa0,ft0,ft0
800000aa:	201085d3          	fsgnj.s	fa1,ft1,ft1
800000ae:	20738653          	fsgnj.s	fa2,ft7,ft7
800000b2:	21ce06d3          	fsgnj.s	fa3,ft8,ft8
800000b6:	2a0d                	c.jal	800001e8 <complex_mul>
800000b8:	20a50153          	fsgnj.s	ft2,fa0,fa0
800000bc:	20b581d3          	fsgnj.s	ft3,fa1,fa1
800000c0:	70040297          	auipc	t0,0x70040
800000c4:	f4028293          	addi	t0,t0,-192 # f0040000 <test_input>
800000c8:	04000e13          	addi	t3,zero,64
800000cc:	03c90eb3          	mul	t4,s2,t3
800000d0:	9ed2                	c.add	t4,s4
800000d2:	0e8a                	c.slli	t4,0x2
800000d4:	92f6                	c.add	t0,t4
800000d6:	0002a207          	flw	ft4,0(t0)
800000da:	3f8002b7          	lui	t0,0x3f800
800000de:	f00282d3          	fmv.w.x	ft5,t0
800000e2:	10527353          	fmul.s	ft6,ft4,ft5
800000e6:	00617153          	fadd.s	ft2,ft2,ft6
800000ea:	006403b3          	add	t2,s0,t1
800000ee:	0023a027          	fsw	ft2,0(t2)
800000f2:	006483b3          	add	t2,s1,t1
800000f6:	0033a027          	fsw	ft3,0(t2)
800000fa:	70040297          	auipc	t0,0x70040
800000fe:	30628293          	addi	t0,t0,774 # f0040400 <test_C_real>
80000102:	006283b3          	add	t2,t0,t1
80000106:	0003a207          	flw	ft4,0(t2)
8000010a:	70040297          	auipc	t0,0x70040
8000010e:	3f628293          	addi	t0,t0,1014 # f0040500 <test_C_imag>
80000112:	006283b3          	add	t2,t0,t1
80000116:	0003a287          	flw	ft5,0(t2)
8000011a:	20420553          	fsgnj.s	fa0,ft4,ft4
8000011e:	205285d3          	fsgnj.s	fa1,ft5,ft5
80000122:	20210653          	fsgnj.s	fa2,ft2,ft2
80000126:	203186d3          	fsgnj.s	fa3,ft3,ft3
8000012a:	287d                	c.jal	800001e8 <complex_mul>
8000012c:	70040297          	auipc	t0,0x70040
80000130:	4d428293          	addi	t0,t0,1236 # f0040600 <test_output>
80000134:	03c90eb3          	mul	t4,s2,t3
80000138:	9ed2                	c.add	t4,s4
8000013a:	0e8a                	c.slli	t4,0x2
8000013c:	92f6                	c.add	t0,t4
8000013e:	00a2a027          	fsw	fa0,0(t0)
80000142:	0a05                	c.addi	s4,1
80000144:	b711                	c.j	80000048 <d_loop>

80000146 <d_done>:
80000146:	0905                	c.addi	s2,1
80000148:	bddd                	c.j	8000003e <t_loop>

8000014a <done>:
8000014a:	d05802b7          	lui	t0,0xd0580
8000014e:	4305                	c.li	t1,1
80000150:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057fdf6>
80000154:	a001                	c.j	80000154 <done+0xa>

80000156 <exp_f>:
80000156:	400002b7          	lui	t0,0x40000
8000015a:	f0028053          	fmv.w.x	ft0,t0
8000015e:	28050553          	fmin.s	fa0,fa0,ft0
80000162:	c00002b7          	lui	t0,0xc0000
80000166:	f0028053          	fmv.w.x	ft0,t0
8000016a:	28051553          	fmax.s	fa0,fa0,ft0
8000016e:	3f8002b7          	lui	t0,0x3f800
80000172:	f0028053          	fmv.w.x	ft0,t0
80000176:	00a07053          	fadd.s	ft0,ft0,fa0
8000017a:	10a570d3          	fmul.s	ft1,fa0,fa0
8000017e:	3f0002b7          	lui	t0,0x3f000
80000182:	f0028153          	fmv.w.x	ft2,t0
80000186:	1020f0d3          	fmul.s	ft1,ft1,ft2
8000018a:	00107053          	fadd.s	ft0,ft0,ft1
8000018e:	10a0f0d3          	fmul.s	ft1,ft1,fa0
80000192:	3e2ab2b7          	lui	t0,0x3e2ab
80000196:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
8000019a:	f0028153          	fmv.w.x	ft2,t0
8000019e:	1020f0d3          	fmul.s	ft1,ft1,ft2
800001a2:	00107553          	fadd.s	fa0,ft0,ft1
800001a6:	8082                	c.jr	ra

800001a8 <sin_f>:
800001a8:	20a50053          	fsgnj.s	ft0,fa0,fa0
800001ac:	10a570d3          	fmul.s	ft1,fa0,fa0
800001b0:	10a0f153          	fmul.s	ft2,ft1,fa0
800001b4:	3e2ab2b7          	lui	t0,0x3e2ab
800001b8:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
800001bc:	f00281d3          	fmv.w.x	ft3,t0
800001c0:	10317153          	fmul.s	ft2,ft2,ft3
800001c4:	08207553          	fsub.s	fa0,ft0,ft2
800001c8:	8082                	c.jr	ra

800001ca <cos_f>:
800001ca:	3f8002b7          	lui	t0,0x3f800
800001ce:	f0028053          	fmv.w.x	ft0,t0
800001d2:	10a570d3          	fmul.s	ft1,fa0,fa0
800001d6:	3f0002b7          	lui	t0,0x3f000
800001da:	f0028153          	fmv.w.x	ft2,t0
800001de:	1020f0d3          	fmul.s	ft1,ft1,ft2
800001e2:	08107553          	fsub.s	fa0,ft0,ft1
800001e6:	8082                	c.jr	ra

800001e8 <complex_mul>:
800001e8:	10c57053          	fmul.s	ft0,fa0,fa2
800001ec:	10d5f0d3          	fmul.s	ft1,fa1,fa3
800001f0:	08107153          	fsub.s	ft2,ft0,ft1
800001f4:	10d57053          	fmul.s	ft0,fa0,fa3
800001f8:	10c5f0d3          	fmul.s	ft1,fa1,fa2
800001fc:	001071d3          	fadd.s	ft3,ft0,ft1
80000200:	20210553          	fsgnj.s	fa0,ft2,ft2
80000204:	203185d3          	fsgnj.s	fa1,ft3,ft3
80000208:	8082                	c.jr	ra
