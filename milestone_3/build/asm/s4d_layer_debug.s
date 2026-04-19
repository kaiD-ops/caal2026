
build/exe/s4d_layer_debug.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70040117          	auipc	sp,0x70040
80000004:	06010113          	addi	sp,sp,96 # f0040060 <_stack_start>
80000008:	70040517          	auipc	a0,0x70040
8000000c:	ff850513          	addi	a0,a0,-8 # f0040000 <test_input>
80000010:	70040597          	auipc	a1,0x70040
80000014:	00058593          	addi	a1,a1,0 # f0040010 <test_A_real>
80000018:	70040617          	auipc	a2,0x70040
8000001c:	00060613          	addi	a2,a2,0 # f0040018 <test_A_imag>
80000020:	70040697          	auipc	a3,0x70040
80000024:	00068693          	addi	a3,a3,0 # f0040020 <test_B>
80000028:	70040717          	auipc	a4,0x70040
8000002c:	00070713          	addi	a4,a4,0 # f0040028 <test_C_real>
80000030:	70040797          	auipc	a5,0x70040
80000034:	00078793          	addi	a5,a5,0 # f0040030 <test_C_imag>
80000038:	70040817          	auipc	a6,0x70040
8000003c:	00080813          	addi	a6,a6,0 # f0040038 <test_output>
80000040:	4889                	c.li	a7,2
80000042:	2039                	c.jal	80000050 <s4d_layer>
80000044:	d05802b7          	lui	t0,0xd0580
80000048:	4305                	c.li	t1,1
8000004a:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057fd7c>
8000004e:	a001                	c.j	8000004e <_start+0x4e>

80000050 <s4d_layer>:
80000050:	715d                	c.addi16sp	sp,-80
80000052:	c686                	c.swsp	ra,76(sp)
80000054:	c4a2                	c.swsp	s0,72(sp)
80000056:	c2a6                	c.swsp	s1,68(sp)
80000058:	c0ca                	c.swsp	s2,64(sp)
8000005a:	de4e                	c.swsp	s3,60(sp)
8000005c:	dc52                	c.swsp	s4,56(sp)
8000005e:	da56                	c.swsp	s5,52(sp)
80000060:	d85a                	c.swsp	s6,48(sp)
80000062:	d65e                	c.swsp	s7,44(sp)
80000064:	d462                	c.swsp	s8,40(sp)
80000066:	d266                	c.swsp	s9,36(sp)
80000068:	f002                	c.fswsp	ft0,32(sp)
8000006a:	ee06                	c.fswsp	ft1,28(sp)
8000006c:	ec0a                	c.fswsp	ft2,24(sp)
8000006e:	ea0e                	c.fswsp	ft3,20(sp)
80000070:	e812                	c.fswsp	ft4,16(sp)
80000072:	e616                	c.fswsp	ft5,12(sp)
80000074:	e41a                	c.fswsp	ft6,8(sp)
80000076:	e21e                	c.fswsp	ft7,4(sp)
80000078:	842a                	c.mv	s0,a0
8000007a:	84ae                	c.mv	s1,a1
8000007c:	8932                	c.mv	s2,a2
8000007e:	89b6                	c.mv	s3,a3
80000080:	8a3a                	c.mv	s4,a4
80000082:	8abe                	c.mv	s5,a5
80000084:	8b42                	c.mv	s6,a6
80000086:	8bc6                	c.mv	s7,a7
80000088:	1101                	c.addi	sp,-32
8000008a:	8d0a                	c.mv	s10,sp
8000008c:	010d0d93          	addi	s11,s10,16
80000090:	000d2023          	sw	zero,0(s10)
80000094:	000d2223          	sw	zero,4(s10)
80000098:	000da023          	sw	zero,0(s11)
8000009c:	000da223          	sw	zero,4(s11)
800000a0:	4c01                	c.li	s8,0

800000a2 <t_loop>:
800000a2:	0d7c5f63          	bge	s8,s7,80000180 <t_done>
800000a6:	4c81                	c.li	s9,0

800000a8 <d_loop>:
800000a8:	4289                	c.li	t0,2
800000aa:	0c5cd963          	bge	s9,t0,8000017c <d_done>
800000ae:	002c9313          	slli	t1,s9,0x2
800000b2:	006483b3          	add	t2,s1,t1
800000b6:	0003a007          	flw	ft0,0(t2)
800000ba:	006903b3          	add	t2,s2,t1
800000be:	0003a087          	flw	ft1,0(t2)
800000c2:	20000553          	fsgnj.s	fa0,ft0,ft0
800000c6:	20dd                	c.jal	800001ac <exp_f>
800000c8:	20a50253          	fsgnj.s	ft4,fa0,fa0
800000cc:	20108553          	fsgnj.s	fa0,ft1,ft1
800000d0:	2a81                	c.jal	80000220 <cos_f>
800000d2:	20a502d3          	fsgnj.s	ft5,fa0,fa0
800000d6:	20108553          	fsgnj.s	fa0,ft1,ft1
800000da:	2215                	c.jal	800001fe <sin_f>
800000dc:	20a50353          	fsgnj.s	ft6,fa0,fa0
800000e0:	105273d3          	fmul.s	ft7,ft4,ft5
800000e4:	10627e53          	fmul.s	ft8,ft4,ft6
800000e8:	002c9313          	slli	t1,s9,0x2
800000ec:	006d03b3          	add	t2,s10,t1
800000f0:	0003a007          	flw	ft0,0(t2)
800000f4:	006d83b3          	add	t2,s11,t1
800000f8:	0003a087          	flw	ft1,0(t2)
800000fc:	20000553          	fsgnj.s	fa0,ft0,ft0
80000100:	201085d3          	fsgnj.s	fa1,ft1,ft1
80000104:	20738653          	fsgnj.s	fa2,ft7,ft7
80000108:	21ce06d3          	fsgnj.s	fa3,ft8,ft8
8000010c:	2a0d                	c.jal	8000023e <complex_mul>
8000010e:	20a50153          	fsgnj.s	ft2,fa0,fa0
80000112:	20b581d3          	fsgnj.s	ft3,fa1,fa1
80000116:	4e09                	c.li	t3,2
80000118:	03cc0eb3          	mul	t4,s8,t3
8000011c:	9ee6                	c.add	t4,s9
8000011e:	0e8a                	c.slli	t4,0x2
80000120:	9ea2                	c.add	t4,s0
80000122:	000ea207          	flw	ft4,0(t4)
80000126:	006983b3          	add	t2,s3,t1
8000012a:	0003a287          	flw	ft5,0(t2)
8000012e:	10527353          	fmul.s	ft6,ft4,ft5
80000132:	00617153          	fadd.s	ft2,ft2,ft6
80000136:	006d03b3          	add	t2,s10,t1
8000013a:	0023a027          	fsw	ft2,0(t2)
8000013e:	006d83b3          	add	t2,s11,t1
80000142:	0033a027          	fsw	ft3,0(t2)
80000146:	006a03b3          	add	t2,s4,t1
8000014a:	0003a207          	flw	ft4,0(t2)
8000014e:	006a83b3          	add	t2,s5,t1
80000152:	0003a287          	flw	ft5,0(t2)
80000156:	20420553          	fsgnj.s	fa0,ft4,ft4
8000015a:	205285d3          	fsgnj.s	fa1,ft5,ft5
8000015e:	20210653          	fsgnj.s	fa2,ft2,ft2
80000162:	203186d3          	fsgnj.s	fa3,ft3,ft3
80000166:	28e1                	c.jal	8000023e <complex_mul>
80000168:	4e09                	c.li	t3,2
8000016a:	03cc0eb3          	mul	t4,s8,t3
8000016e:	9ee6                	c.add	t4,s9
80000170:	0e8a                	c.slli	t4,0x2
80000172:	9eda                	c.add	t4,s6
80000174:	00aea027          	fsw	fa0,0(t4)
80000178:	0c85                	c.addi	s9,1
8000017a:	b73d                	c.j	800000a8 <d_loop>

8000017c <d_done>:
8000017c:	0c05                	c.addi	s8,1
8000017e:	b715                	c.j	800000a2 <t_loop>

80000180 <t_done>:
80000180:	7002                	c.flwsp	ft0,32(sp)
80000182:	60f2                	c.flwsp	ft1,28(sp)
80000184:	6162                	c.flwsp	ft2,24(sp)
80000186:	61d2                	c.flwsp	ft3,20(sp)
80000188:	6242                	c.flwsp	ft4,16(sp)
8000018a:	62b2                	c.flwsp	ft5,12(sp)
8000018c:	6322                	c.flwsp	ft6,8(sp)
8000018e:	6392                	c.flwsp	ft7,4(sp)
80000190:	6105                	c.addi16sp	sp,32
80000192:	40b6                	c.lwsp	ra,76(sp)
80000194:	4426                	c.lwsp	s0,72(sp)
80000196:	4496                	c.lwsp	s1,68(sp)
80000198:	4906                	c.lwsp	s2,64(sp)
8000019a:	59f2                	c.lwsp	s3,60(sp)
8000019c:	5a62                	c.lwsp	s4,56(sp)
8000019e:	5ad2                	c.lwsp	s5,52(sp)
800001a0:	5b42                	c.lwsp	s6,48(sp)
800001a2:	5bb2                	c.lwsp	s7,44(sp)
800001a4:	5c22                	c.lwsp	s8,40(sp)
800001a6:	5c92                	c.lwsp	s9,36(sp)
800001a8:	6161                	c.addi16sp	sp,80
800001aa:	8082                	c.jr	ra

800001ac <exp_f>:
800001ac:	400002b7          	lui	t0,0x40000
800001b0:	f0028053          	fmv.w.x	ft0,t0
800001b4:	28050553          	fmin.s	fa0,fa0,ft0
800001b8:	c00002b7          	lui	t0,0xc0000
800001bc:	f0028053          	fmv.w.x	ft0,t0
800001c0:	28051553          	fmax.s	fa0,fa0,ft0
800001c4:	3f8002b7          	lui	t0,0x3f800
800001c8:	f0028053          	fmv.w.x	ft0,t0
800001cc:	00a07053          	fadd.s	ft0,ft0,fa0
800001d0:	10a570d3          	fmul.s	ft1,fa0,fa0
800001d4:	3f0002b7          	lui	t0,0x3f000
800001d8:	f0028153          	fmv.w.x	ft2,t0
800001dc:	1020f0d3          	fmul.s	ft1,ft1,ft2
800001e0:	00107053          	fadd.s	ft0,ft0,ft1
800001e4:	10a0f0d3          	fmul.s	ft1,ft1,fa0
800001e8:	3e2ab2b7          	lui	t0,0x3e2ab
800001ec:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
800001f0:	f0028153          	fmv.w.x	ft2,t0
800001f4:	1020f0d3          	fmul.s	ft1,ft1,ft2
800001f8:	00107553          	fadd.s	fa0,ft0,ft1
800001fc:	8082                	c.jr	ra

800001fe <sin_f>:
800001fe:	20a50053          	fsgnj.s	ft0,fa0,fa0
80000202:	10a570d3          	fmul.s	ft1,fa0,fa0
80000206:	10a0f153          	fmul.s	ft2,ft1,fa0
8000020a:	3e2ab2b7          	lui	t0,0x3e2ab
8000020e:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
80000212:	f00281d3          	fmv.w.x	ft3,t0
80000216:	10317153          	fmul.s	ft2,ft2,ft3
8000021a:	08207553          	fsub.s	fa0,ft0,ft2
8000021e:	8082                	c.jr	ra

80000220 <cos_f>:
80000220:	3f8002b7          	lui	t0,0x3f800
80000224:	f0028053          	fmv.w.x	ft0,t0
80000228:	10a570d3          	fmul.s	ft1,fa0,fa0
8000022c:	3f0002b7          	lui	t0,0x3f000
80000230:	f0028153          	fmv.w.x	ft2,t0
80000234:	1020f0d3          	fmul.s	ft1,ft1,ft2
80000238:	08107553          	fsub.s	fa0,ft0,ft1
8000023c:	8082                	c.jr	ra

8000023e <complex_mul>:
8000023e:	10c57053          	fmul.s	ft0,fa0,fa2
80000242:	10d5f0d3          	fmul.s	ft1,fa1,fa3
80000246:	08107153          	fsub.s	ft2,ft0,ft1
8000024a:	10d57053          	fmul.s	ft0,fa0,fa3
8000024e:	10c5f0d3          	fmul.s	ft1,fa1,fa2
80000252:	001071d3          	fadd.s	ft3,ft0,ft1
80000256:	20210553          	fsgnj.s	fa0,ft2,ft2
8000025a:	203185d3          	fsgnj.s	fa1,ft3,ft3
8000025e:	8082                	c.jr	ra

80000260 <tanh_f>:
80000260:	1141                	c.addi	sp,-16
80000262:	c606                	c.swsp	ra,12(sp)
80000264:	00a57553          	fadd.s	fa0,fa0,fa0
80000268:	3791                	c.jal	800001ac <exp_f>
8000026a:	3f8002b7          	lui	t0,0x3f800
8000026e:	f0028053          	fmv.w.x	ft0,t0
80000272:	080575d3          	fsub.s	fa1,fa0,ft0
80000276:	00057653          	fadd.s	fa2,fa0,ft0
8000027a:	18c5f553          	fdiv.s	fa0,fa1,fa2
8000027e:	40b2                	c.lwsp	ra,12(sp)
80000280:	0141                	c.addi	sp,16
80000282:	8082                	c.jr	ra
