
build/exe/s4d_layer_full.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70041117          	auipc	sp,0x70041
80000004:	90010113          	addi	sp,sp,-1792 # f0040900 <_stack_start>
80000008:	70040517          	auipc	a0,0x70040
8000000c:	ff850513          	addi	a0,a0,-8 # f0040000 <test_input>
80000010:	70040597          	auipc	a1,0x70040
80000014:	1f058593          	addi	a1,a1,496 # f0040200 <test_A_real>
80000018:	70040617          	auipc	a2,0x70040
8000001c:	2e860613          	addi	a2,a2,744 # f0040300 <test_A_imag>
80000020:	70040697          	auipc	a3,0x70040
80000024:	3e068693          	addi	a3,a3,992 # f0040400 <test_B>
80000028:	70040717          	auipc	a4,0x70040
8000002c:	4d870713          	addi	a4,a4,1240 # f0040500 <test_C_real>
80000030:	70040797          	auipc	a5,0x70040
80000034:	5d078793          	addi	a5,a5,1488 # f0040600 <test_C_imag>
80000038:	70040817          	auipc	a6,0x70040
8000003c:	6c880813          	addi	a6,a6,1736 # f0040700 <test_output>
80000040:	4889                	c.li	a7,2
80000042:	2039                	c.jal	80000050 <s4d_layer>
80000044:	d05802b7          	lui	t0,0xd0580
80000048:	4305                	c.li	t1,1
8000004a:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057fd66>
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
80000068:	d06a                	c.swsp	s10,32(sp)
8000006a:	ce6e                	c.swsp	s11,28(sp)
8000006c:	ec02                	c.fswsp	ft0,24(sp)
8000006e:	ea06                	c.fswsp	ft1,20(sp)
80000070:	e80a                	c.fswsp	ft2,16(sp)
80000072:	e60e                	c.fswsp	ft3,12(sp)
80000074:	e412                	c.fswsp	ft4,8(sp)
80000076:	e216                	c.fswsp	ft5,4(sp)
80000078:	842a                	c.mv	s0,a0
8000007a:	84ae                	c.mv	s1,a1
8000007c:	8932                	c.mv	s2,a2
8000007e:	89b6                	c.mv	s3,a3
80000080:	8a3a                	c.mv	s4,a4
80000082:	8abe                	c.mv	s5,a5
80000084:	8b42                	c.mv	s6,a6
80000086:	8bc6                	c.mv	s7,a7
80000088:	7101                	c.addi16sp	sp,-512
8000008a:	8d0a                	c.mv	s10,sp
8000008c:	100d0d93          	addi	s11,s10,256
80000090:	4281                	c.li	t0,0

80000092 <zero_loop>:
80000092:	04000313          	addi	t1,zero,64
80000096:	0062de63          	bge	t0,t1,800000b2 <zero_done>
8000009a:	00229393          	slli	t2,t0,0x2
8000009e:	007d0e33          	add	t3,s10,t2
800000a2:	000e2023          	sw	zero,0(t3)
800000a6:	007d8e33          	add	t3,s11,t2
800000aa:	000e2023          	sw	zero,0(t3)
800000ae:	0285                	c.addi	t0,1
800000b0:	b7cd                	c.j	80000092 <zero_loop>

800000b2 <zero_done>:
800000b2:	4c01                	c.li	s8,0

800000b4 <t_loop>:
800000b4:	0f7c5063          	bge	s8,s7,80000194 <t_done>
800000b8:	4c81                	c.li	s9,0

800000ba <d_loop>:
800000ba:	04000293          	addi	t0,zero,64
800000be:	0c5cd963          	bge	s9,t0,80000190 <d_done>
800000c2:	002c9313          	slli	t1,s9,0x2
800000c6:	006483b3          	add	t2,s1,t1
800000ca:	0003a007          	flw	ft0,0(t2)
800000ce:	006903b3          	add	t2,s2,t1
800000d2:	0003a087          	flw	ft1,0(t2)
800000d6:	20000553          	fsgnj.s	fa0,ft0,ft0
800000da:	20e5                	c.jal	800001c2 <exp_f>
800000dc:	20a50253          	fsgnj.s	ft4,fa0,fa0
800000e0:	20108553          	fsgnj.s	fa0,ft1,ft1
800000e4:	2a89                	c.jal	80000236 <cos_f>
800000e6:	20a502d3          	fsgnj.s	ft5,fa0,fa0
800000ea:	20108553          	fsgnj.s	fa0,ft1,ft1
800000ee:	221d                	c.jal	80000214 <sin_f>
800000f0:	20a50353          	fsgnj.s	ft6,fa0,fa0
800000f4:	105273d3          	fmul.s	ft7,ft4,ft5
800000f8:	10627e53          	fmul.s	ft8,ft4,ft6
800000fc:	006d03b3          	add	t2,s10,t1
80000100:	0003a007          	flw	ft0,0(t2)
80000104:	006d83b3          	add	t2,s11,t1
80000108:	0003a087          	flw	ft1,0(t2)
8000010c:	20000553          	fsgnj.s	fa0,ft0,ft0
80000110:	201085d3          	fsgnj.s	fa1,ft1,ft1
80000114:	20738653          	fsgnj.s	fa2,ft7,ft7
80000118:	21ce06d3          	fsgnj.s	fa3,ft8,ft8
8000011c:	2a25                	c.jal	80000254 <complex_mul>
8000011e:	20a50153          	fsgnj.s	ft2,fa0,fa0
80000122:	20b581d3          	fsgnj.s	ft3,fa1,fa1
80000126:	04000e13          	addi	t3,zero,64
8000012a:	03cc0eb3          	mul	t4,s8,t3
8000012e:	9ee6                	c.add	t4,s9
80000130:	0e8a                	c.slli	t4,0x2
80000132:	9ea2                	c.add	t4,s0
80000134:	000ea207          	flw	ft4,0(t4)
80000138:	006983b3          	add	t2,s3,t1
8000013c:	0003a287          	flw	ft5,0(t2)
80000140:	10527353          	fmul.s	ft6,ft4,ft5
80000144:	00617153          	fadd.s	ft2,ft2,ft6
80000148:	006d03b3          	add	t2,s10,t1
8000014c:	0023a027          	fsw	ft2,0(t2)
80000150:	006d83b3          	add	t2,s11,t1
80000154:	0033a027          	fsw	ft3,0(t2)
80000158:	006a03b3          	add	t2,s4,t1
8000015c:	0003a207          	flw	ft4,0(t2)
80000160:	006a83b3          	add	t2,s5,t1
80000164:	0003a287          	flw	ft5,0(t2)
80000168:	20420553          	fsgnj.s	fa0,ft4,ft4
8000016c:	205285d3          	fsgnj.s	fa1,ft5,ft5
80000170:	20210653          	fsgnj.s	fa2,ft2,ft2
80000174:	203186d3          	fsgnj.s	fa3,ft3,ft3
80000178:	28f1                	c.jal	80000254 <complex_mul>
8000017a:	04000e13          	addi	t3,zero,64
8000017e:	03cc0eb3          	mul	t4,s8,t3
80000182:	9ee6                	c.add	t4,s9
80000184:	0e8a                	c.slli	t4,0x2
80000186:	9eda                	c.add	t4,s6
80000188:	00aea027          	fsw	fa0,0(t4)
8000018c:	0c85                	c.addi	s9,1
8000018e:	b735                	c.j	800000ba <d_loop>

80000190 <d_done>:
80000190:	0c05                	c.addi	s8,1
80000192:	b70d                	c.j	800000b4 <t_loop>

80000194 <t_done>:
80000194:	20010113          	addi	sp,sp,512
80000198:	6062                	c.flwsp	ft0,24(sp)
8000019a:	60d2                	c.flwsp	ft1,20(sp)
8000019c:	6142                	c.flwsp	ft2,16(sp)
8000019e:	61b2                	c.flwsp	ft3,12(sp)
800001a0:	6222                	c.flwsp	ft4,8(sp)
800001a2:	6292                	c.flwsp	ft5,4(sp)
800001a4:	40b6                	c.lwsp	ra,76(sp)
800001a6:	4426                	c.lwsp	s0,72(sp)
800001a8:	4496                	c.lwsp	s1,68(sp)
800001aa:	4906                	c.lwsp	s2,64(sp)
800001ac:	59f2                	c.lwsp	s3,60(sp)
800001ae:	5a62                	c.lwsp	s4,56(sp)
800001b0:	5ad2                	c.lwsp	s5,52(sp)
800001b2:	5b42                	c.lwsp	s6,48(sp)
800001b4:	5bb2                	c.lwsp	s7,44(sp)
800001b6:	5c22                	c.lwsp	s8,40(sp)
800001b8:	5c92                	c.lwsp	s9,36(sp)
800001ba:	5d02                	c.lwsp	s10,32(sp)
800001bc:	4df2                	c.lwsp	s11,28(sp)
800001be:	6161                	c.addi16sp	sp,80
800001c0:	8082                	c.jr	ra

800001c2 <exp_f>:
800001c2:	400002b7          	lui	t0,0x40000
800001c6:	f0028053          	fmv.w.x	ft0,t0
800001ca:	28050553          	fmin.s	fa0,fa0,ft0
800001ce:	c00002b7          	lui	t0,0xc0000
800001d2:	f0028053          	fmv.w.x	ft0,t0
800001d6:	28051553          	fmax.s	fa0,fa0,ft0
800001da:	3f8002b7          	lui	t0,0x3f800
800001de:	f0028053          	fmv.w.x	ft0,t0
800001e2:	00a07053          	fadd.s	ft0,ft0,fa0
800001e6:	10a570d3          	fmul.s	ft1,fa0,fa0
800001ea:	3f0002b7          	lui	t0,0x3f000
800001ee:	f0028153          	fmv.w.x	ft2,t0
800001f2:	1020f0d3          	fmul.s	ft1,ft1,ft2
800001f6:	00107053          	fadd.s	ft0,ft0,ft1
800001fa:	10a0f0d3          	fmul.s	ft1,ft1,fa0
800001fe:	3e2ab2b7          	lui	t0,0x3e2ab
80000202:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
80000206:	f0028153          	fmv.w.x	ft2,t0
8000020a:	1020f0d3          	fmul.s	ft1,ft1,ft2
8000020e:	00107553          	fadd.s	fa0,ft0,ft1
80000212:	8082                	c.jr	ra

80000214 <sin_f>:
80000214:	20a50053          	fsgnj.s	ft0,fa0,fa0
80000218:	10a570d3          	fmul.s	ft1,fa0,fa0
8000021c:	10a0f153          	fmul.s	ft2,ft1,fa0
80000220:	3e2ab2b7          	lui	t0,0x3e2ab
80000224:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
80000228:	f00281d3          	fmv.w.x	ft3,t0
8000022c:	10317153          	fmul.s	ft2,ft2,ft3
80000230:	08207553          	fsub.s	fa0,ft0,ft2
80000234:	8082                	c.jr	ra

80000236 <cos_f>:
80000236:	3f8002b7          	lui	t0,0x3f800
8000023a:	f0028053          	fmv.w.x	ft0,t0
8000023e:	10a570d3          	fmul.s	ft1,fa0,fa0
80000242:	3f0002b7          	lui	t0,0x3f000
80000246:	f0028153          	fmv.w.x	ft2,t0
8000024a:	1020f0d3          	fmul.s	ft1,ft1,ft2
8000024e:	08107553          	fsub.s	fa0,ft0,ft1
80000252:	8082                	c.jr	ra

80000254 <complex_mul>:
80000254:	10c57053          	fmul.s	ft0,fa0,fa2
80000258:	10d5f0d3          	fmul.s	ft1,fa1,fa3
8000025c:	08107153          	fsub.s	ft2,ft0,ft1
80000260:	10d57053          	fmul.s	ft0,fa0,fa3
80000264:	10c5f0d3          	fmul.s	ft1,fa1,fa2
80000268:	001071d3          	fadd.s	ft3,ft0,ft1
8000026c:	20210553          	fsgnj.s	fa0,ft2,ft2
80000270:	203185d3          	fsgnj.s	fa1,ft3,ft3
80000274:	8082                	c.jr	ra

80000276 <tanh_f>:
80000276:	1141                	c.addi	sp,-16
80000278:	c606                	c.swsp	ra,12(sp)
8000027a:	00a57553          	fadd.s	fa0,fa0,fa0
8000027e:	3791                	c.jal	800001c2 <exp_f>
80000280:	3f8002b7          	lui	t0,0x3f800
80000284:	f0028053          	fmv.w.x	ft0,t0
80000288:	080575d3          	fsub.s	fa1,fa0,ft0
8000028c:	00057653          	fadd.s	fa2,fa0,ft0
80000290:	18c5f553          	fdiv.s	fa0,fa1,fa2
80000294:	40b2                	c.lwsp	ra,12(sp)
80000296:	0141                	c.addi	sp,16
80000298:	8082                	c.jr	ra
