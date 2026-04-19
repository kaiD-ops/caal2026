
build/exe/test_d_model_8.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70040117          	auipc	sp,0x70040
80000004:	14010113          	addi	sp,sp,320 # f0040140 <_stack_start>
80000008:	70040417          	auipc	s0,0x70040
8000000c:	0f840413          	addi	s0,s0,248 # f0040100 <h_real>
80000010:	70040497          	auipc	s1,0x70040
80000014:	11048493          	addi	s1,s1,272 # f0040120 <h_imag>
80000018:	4281                	c.li	t0,0

8000001a <init_loop>:
8000001a:	4321                	c.li	t1,8
8000001c:	0062de63          	bge	t0,t1,80000038 <init_done>
80000020:	00229393          	slli	t2,t0,0x2
80000024:	00740e33          	add	t3,s0,t2
80000028:	000e2023          	sw	zero,0(t3)
8000002c:	00748e33          	add	t3,s1,t2
80000030:	000e2023          	sw	zero,0(t3)
80000034:	0285                	c.addi	t0,1
80000036:	b7d5                	c.j	8000001a <init_loop>

80000038 <init_done>:
80000038:	4901                	c.li	s2,0
8000003a:	4989                	c.li	s3,2

8000003c <t_loop>:
8000003c:	11395463          	bge	s2,s3,80000144 <done>
80000040:	4a01                	c.li	s4,0
80000042:	4aa1                	c.li	s5,8

80000044 <d_loop>:
80000044:	0f5a5e63          	bge	s4,s5,80000140 <d_done>
80000048:	70040297          	auipc	t0,0x70040
8000004c:	ff828293          	addi	t0,t0,-8 # f0040040 <test_A_real>
80000050:	002a1313          	slli	t1,s4,0x2
80000054:	006283b3          	add	t2,t0,t1
80000058:	0003a007          	flw	ft0,0(t2)
8000005c:	70040297          	auipc	t0,0x70040
80000060:	00428293          	addi	t0,t0,4 # f0040060 <test_A_imag>
80000064:	006283b3          	add	t2,t0,t1
80000068:	0003a087          	flw	ft1,0(t2)
8000006c:	20000553          	fsgnj.s	fa0,ft0,ft0
80000070:	20c5                	c.jal	80000150 <exp_f>
80000072:	20a50253          	fsgnj.s	ft4,fa0,fa0
80000076:	20108553          	fsgnj.s	fa0,ft1,ft1
8000007a:	22a9                	c.jal	800001c4 <cos_f>
8000007c:	20a502d3          	fsgnj.s	ft5,fa0,fa0
80000080:	20108553          	fsgnj.s	fa0,ft1,ft1
80000084:	2a39                	c.jal	800001a2 <sin_f>
80000086:	20a50353          	fsgnj.s	ft6,fa0,fa0
8000008a:	105273d3          	fmul.s	ft7,ft4,ft5
8000008e:	10627e53          	fmul.s	ft8,ft4,ft6
80000092:	006403b3          	add	t2,s0,t1
80000096:	0003a007          	flw	ft0,0(t2)
8000009a:	006483b3          	add	t2,s1,t1
8000009e:	0003a087          	flw	ft1,0(t2)
800000a2:	20000553          	fsgnj.s	fa0,ft0,ft0
800000a6:	201085d3          	fsgnj.s	fa1,ft1,ft1
800000aa:	20738653          	fsgnj.s	fa2,ft7,ft7
800000ae:	21ce06d3          	fsgnj.s	fa3,ft8,ft8
800000b2:	2a05                	c.jal	800001e2 <complex_mul>
800000b4:	20a50153          	fsgnj.s	ft2,fa0,fa0
800000b8:	20b581d3          	fsgnj.s	ft3,fa1,fa1
800000bc:	70040297          	auipc	t0,0x70040
800000c0:	f4428293          	addi	t0,t0,-188 # f0040000 <test_input>
800000c4:	4e21                	c.li	t3,8
800000c6:	03c90eb3          	mul	t4,s2,t3
800000ca:	9ed2                	c.add	t4,s4
800000cc:	0e8a                	c.slli	t4,0x2
800000ce:	92f6                	c.add	t0,t4
800000d0:	0002a207          	flw	ft4,0(t0)
800000d4:	3f8002b7          	lui	t0,0x3f800
800000d8:	f00282d3          	fmv.w.x	ft5,t0
800000dc:	10527353          	fmul.s	ft6,ft4,ft5
800000e0:	00617153          	fadd.s	ft2,ft2,ft6
800000e4:	006403b3          	add	t2,s0,t1
800000e8:	0023a027          	fsw	ft2,0(t2)
800000ec:	006483b3          	add	t2,s1,t1
800000f0:	0033a027          	fsw	ft3,0(t2)
800000f4:	70040297          	auipc	t0,0x70040
800000f8:	f8c28293          	addi	t0,t0,-116 # f0040080 <test_C_real>
800000fc:	006283b3          	add	t2,t0,t1
80000100:	0003a207          	flw	ft4,0(t2)
80000104:	70040297          	auipc	t0,0x70040
80000108:	f9c28293          	addi	t0,t0,-100 # f00400a0 <test_C_imag>
8000010c:	006283b3          	add	t2,t0,t1
80000110:	0003a287          	flw	ft5,0(t2)
80000114:	20420553          	fsgnj.s	fa0,ft4,ft4
80000118:	205285d3          	fsgnj.s	fa1,ft5,ft5
8000011c:	20210653          	fsgnj.s	fa2,ft2,ft2
80000120:	203186d3          	fsgnj.s	fa3,ft3,ft3
80000124:	287d                	c.jal	800001e2 <complex_mul>
80000126:	70040297          	auipc	t0,0x70040
8000012a:	f9a28293          	addi	t0,t0,-102 # f00400c0 <test_output>
8000012e:	03c90eb3          	mul	t4,s2,t3
80000132:	9ed2                	c.add	t4,s4
80000134:	0e8a                	c.slli	t4,0x2
80000136:	92f6                	c.add	t0,t4
80000138:	00a2a027          	fsw	fa0,0(t0)
8000013c:	0a05                	c.addi	s4,1
8000013e:	b719                	c.j	80000044 <d_loop>

80000140 <d_done>:
80000140:	0905                	c.addi	s2,1
80000142:	bded                	c.j	8000003c <t_loop>

80000144 <done>:
80000144:	d05802b7          	lui	t0,0xd0580
80000148:	4305                	c.li	t1,1
8000014a:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057fdfc>
8000014e:	a001                	c.j	8000014e <done+0xa>

80000150 <exp_f>:
80000150:	400002b7          	lui	t0,0x40000
80000154:	f0028053          	fmv.w.x	ft0,t0
80000158:	28050553          	fmin.s	fa0,fa0,ft0
8000015c:	c00002b7          	lui	t0,0xc0000
80000160:	f0028053          	fmv.w.x	ft0,t0
80000164:	28051553          	fmax.s	fa0,fa0,ft0
80000168:	3f8002b7          	lui	t0,0x3f800
8000016c:	f0028053          	fmv.w.x	ft0,t0
80000170:	00a07053          	fadd.s	ft0,ft0,fa0
80000174:	10a570d3          	fmul.s	ft1,fa0,fa0
80000178:	3f0002b7          	lui	t0,0x3f000
8000017c:	f0028153          	fmv.w.x	ft2,t0
80000180:	1020f0d3          	fmul.s	ft1,ft1,ft2
80000184:	00107053          	fadd.s	ft0,ft0,ft1
80000188:	10a0f0d3          	fmul.s	ft1,ft1,fa0
8000018c:	3e2ab2b7          	lui	t0,0x3e2ab
80000190:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
80000194:	f0028153          	fmv.w.x	ft2,t0
80000198:	1020f0d3          	fmul.s	ft1,ft1,ft2
8000019c:	00107553          	fadd.s	fa0,ft0,ft1
800001a0:	8082                	c.jr	ra

800001a2 <sin_f>:
800001a2:	20a50053          	fsgnj.s	ft0,fa0,fa0
800001a6:	10a570d3          	fmul.s	ft1,fa0,fa0
800001aa:	10a0f153          	fmul.s	ft2,ft1,fa0
800001ae:	3e2ab2b7          	lui	t0,0x3e2ab
800001b2:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
800001b6:	f00281d3          	fmv.w.x	ft3,t0
800001ba:	10317153          	fmul.s	ft2,ft2,ft3
800001be:	08207553          	fsub.s	fa0,ft0,ft2
800001c2:	8082                	c.jr	ra

800001c4 <cos_f>:
800001c4:	3f8002b7          	lui	t0,0x3f800
800001c8:	f0028053          	fmv.w.x	ft0,t0
800001cc:	10a570d3          	fmul.s	ft1,fa0,fa0
800001d0:	3f0002b7          	lui	t0,0x3f000
800001d4:	f0028153          	fmv.w.x	ft2,t0
800001d8:	1020f0d3          	fmul.s	ft1,ft1,ft2
800001dc:	08107553          	fsub.s	fa0,ft0,ft1
800001e0:	8082                	c.jr	ra

800001e2 <complex_mul>:
800001e2:	10c57053          	fmul.s	ft0,fa0,fa2
800001e6:	10d5f0d3          	fmul.s	ft1,fa1,fa3
800001ea:	08107153          	fsub.s	ft2,ft0,ft1
800001ee:	10d57053          	fmul.s	ft0,fa0,fa3
800001f2:	10c5f0d3          	fmul.s	ft1,fa1,fa2
800001f6:	001071d3          	fadd.s	ft3,ft0,ft1
800001fa:	20210553          	fsgnj.s	fa0,ft2,ft2
800001fe:	203185d3          	fsgnj.s	fa1,ft3,ft3
80000202:	8082                	c.jr	ra
