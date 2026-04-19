
build/exe/s4d_layer.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70040117          	auipc	sp,0x70040
80000004:	26010113          	addi	sp,sp,608 # f0040260 <_stack_start>
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
8000004a:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057fd42>
8000004e:	a001                	c.j	8000004e <_start+0x4e>

80000050 <s4d_layer>:
80000050:	711d                	c.addi16sp	sp,-96
80000052:	ce86                	c.swsp	ra,92(sp)
80000054:	cca2                	c.swsp	s0,88(sp)
80000056:	caa6                	c.swsp	s1,84(sp)
80000058:	c8ca                	c.swsp	s2,80(sp)
8000005a:	c6ce                	c.swsp	s3,76(sp)
8000005c:	c4d2                	c.swsp	s4,72(sp)
8000005e:	c2d6                	c.swsp	s5,68(sp)
80000060:	c0da                	c.swsp	s6,64(sp)
80000062:	de5e                	c.swsp	s7,60(sp)
80000064:	dc62                	c.swsp	s8,56(sp)
80000066:	da66                	c.swsp	s9,52(sp)
80000068:	d86a                	c.swsp	s10,48(sp)
8000006a:	d66e                	c.swsp	s11,44(sp)
8000006c:	f402                	c.fswsp	ft0,40(sp)
8000006e:	f206                	c.fswsp	ft1,36(sp)
80000070:	f00a                	c.fswsp	ft2,32(sp)
80000072:	ee0e                	c.fswsp	ft3,28(sp)
80000074:	ec12                	c.fswsp	ft4,24(sp)
80000076:	ea16                	c.fswsp	ft5,20(sp)
80000078:	e81a                	c.fswsp	ft6,16(sp)
8000007a:	e61e                	c.fswsp	ft7,12(sp)
8000007c:	842a                	c.mv	s0,a0
8000007e:	84ae                	c.mv	s1,a1
80000080:	8932                	c.mv	s2,a2
80000082:	89b6                	c.mv	s3,a3
80000084:	8a3a                	c.mv	s4,a4
80000086:	8abe                	c.mv	s5,a5
80000088:	8b42                	c.mv	s6,a6
8000008a:	8bc6                	c.mv	s7,a7
8000008c:	70040d17          	auipc	s10,0x70040
80000090:	fd4d0d13          	addi	s10,s10,-44 # f0040060 <s4d_h_real>
80000094:	70040d97          	auipc	s11,0x70040
80000098:	0ccd8d93          	addi	s11,s11,204 # f0040160 <s4d_h_imag>
8000009c:	4281                	c.li	t0,0

8000009e <zero_state>:
8000009e:	04000313          	addi	t1,zero,64
800000a2:	0062de63          	bge	t0,t1,800000be <zero_done>
800000a6:	00229393          	slli	t2,t0,0x2
800000aa:	007d0e33          	add	t3,s10,t2
800000ae:	000e2023          	sw	zero,0(t3)
800000b2:	007d8e33          	add	t3,s11,t2
800000b6:	000e2023          	sw	zero,0(t3)
800000ba:	0285                	c.addi	t0,1
800000bc:	b7cd                	c.j	8000009e <zero_state>

800000be <zero_done>:
800000be:	4c01                	c.li	s8,0

800000c0 <t_loop>:
800000c0:	0f7c5c63          	bge	s8,s7,800001b8 <t_done>
800000c4:	4c81                	c.li	s9,0

800000c6 <d_loop>:
800000c6:	04000293          	addi	t0,zero,64
800000ca:	0e5cd563          	bge	s9,t0,800001b4 <d_done>
800000ce:	002c9313          	slli	t1,s9,0x2
800000d2:	006483b3          	add	t2,s1,t1
800000d6:	0003a007          	flw	ft0,0(t2)
800000da:	006903b3          	add	t2,s2,t1
800000de:	0003a087          	flw	ft1,0(t2)
800000e2:	20000153          	fsgnj.s	ft2,ft0,ft0
800000e6:	201081d3          	fsgnj.s	ft3,ft1,ft1
800000ea:	20210553          	fsgnj.s	fa0,ft2,ft2
800000ee:	28e5                	c.jal	800001e6 <exp_f>
800000f0:	20a50253          	fsgnj.s	ft4,fa0,fa0
800000f4:	20318553          	fsgnj.s	fa0,ft3,ft3
800000f8:	228d                	c.jal	8000025a <cos_f>
800000fa:	20a502d3          	fsgnj.s	ft5,fa0,fa0
800000fe:	20318553          	fsgnj.s	fa0,ft3,ft3
80000102:	2a1d                	c.jal	80000238 <sin_f>
80000104:	20a50353          	fsgnj.s	ft6,fa0,fa0
80000108:	105273d3          	fmul.s	ft7,ft4,ft5
8000010c:	10627e53          	fmul.s	ft8,ft4,ft6
80000110:	002c9313          	slli	t1,s9,0x2
80000114:	006d03b3          	add	t2,s10,t1
80000118:	0003a007          	flw	ft0,0(t2)
8000011c:	006d83b3          	add	t2,s11,t1
80000120:	0003a087          	flw	ft1,0(t2)
80000124:	20000553          	fsgnj.s	fa0,ft0,ft0
80000128:	201085d3          	fsgnj.s	fa1,ft1,ft1
8000012c:	20738653          	fsgnj.s	fa2,ft7,ft7
80000130:	21ce06d3          	fsgnj.s	fa3,ft8,ft8
80000134:	2291                	c.jal	80000278 <complex_mul>
80000136:	20a50153          	fsgnj.s	ft2,fa0,fa0
8000013a:	20b581d3          	fsgnj.s	ft3,fa1,fa1
8000013e:	04000e13          	addi	t3,zero,64
80000142:	03cc0eb3          	mul	t4,s8,t3
80000146:	9ee6                	c.add	t4,s9
80000148:	0e8a                	c.slli	t4,0x2
8000014a:	9ea2                	c.add	t4,s0
8000014c:	000ea207          	flw	ft4,0(t4)
80000150:	002c9313          	slli	t1,s9,0x2
80000154:	006983b3          	add	t2,s3,t1
80000158:	0003a287          	flw	ft5,0(t2)
8000015c:	10527353          	fmul.s	ft6,ft4,ft5
80000160:	00617153          	fadd.s	ft2,ft2,ft6
80000164:	002c9313          	slli	t1,s9,0x2
80000168:	006d03b3          	add	t2,s10,t1
8000016c:	0023a027          	fsw	ft2,0(t2)
80000170:	006d83b3          	add	t2,s11,t1
80000174:	0033a027          	fsw	ft3,0(t2)
80000178:	002c9313          	slli	t1,s9,0x2
8000017c:	006a03b3          	add	t2,s4,t1
80000180:	0003a207          	flw	ft4,0(t2)
80000184:	006a83b3          	add	t2,s5,t1
80000188:	0003a287          	flw	ft5,0(t2)
8000018c:	20420553          	fsgnj.s	fa0,ft4,ft4
80000190:	205285d3          	fsgnj.s	fa1,ft5,ft5
80000194:	20210653          	fsgnj.s	fa2,ft2,ft2
80000198:	203186d3          	fsgnj.s	fa3,ft3,ft3
8000019c:	28f1                	c.jal	80000278 <complex_mul>
8000019e:	04000e13          	addi	t3,zero,64
800001a2:	03cc0eb3          	mul	t4,s8,t3
800001a6:	9ee6                	c.add	t4,s9
800001a8:	0e8a                	c.slli	t4,0x2
800001aa:	9eda                	c.add	t4,s6
800001ac:	00aea027          	fsw	fa0,0(t4)
800001b0:	0c85                	c.addi	s9,1
800001b2:	bf11                	c.j	800000c6 <d_loop>

800001b4 <d_done>:
800001b4:	0c05                	c.addi	s8,1
800001b6:	b729                	c.j	800000c0 <t_loop>

800001b8 <t_done>:
800001b8:	7022                	c.flwsp	ft0,40(sp)
800001ba:	7092                	c.flwsp	ft1,36(sp)
800001bc:	7102                	c.flwsp	ft2,32(sp)
800001be:	61f2                	c.flwsp	ft3,28(sp)
800001c0:	6262                	c.flwsp	ft4,24(sp)
800001c2:	62d2                	c.flwsp	ft5,20(sp)
800001c4:	6342                	c.flwsp	ft6,16(sp)
800001c6:	63b2                	c.flwsp	ft7,12(sp)
800001c8:	40f6                	c.lwsp	ra,92(sp)
800001ca:	4466                	c.lwsp	s0,88(sp)
800001cc:	44d6                	c.lwsp	s1,84(sp)
800001ce:	4946                	c.lwsp	s2,80(sp)
800001d0:	49b6                	c.lwsp	s3,76(sp)
800001d2:	4a26                	c.lwsp	s4,72(sp)
800001d4:	4a96                	c.lwsp	s5,68(sp)
800001d6:	4b06                	c.lwsp	s6,64(sp)
800001d8:	5bf2                	c.lwsp	s7,60(sp)
800001da:	5c62                	c.lwsp	s8,56(sp)
800001dc:	5cd2                	c.lwsp	s9,52(sp)
800001de:	5d42                	c.lwsp	s10,48(sp)
800001e0:	5db2                	c.lwsp	s11,44(sp)
800001e2:	6125                	c.addi16sp	sp,96
800001e4:	8082                	c.jr	ra

800001e6 <exp_f>:
800001e6:	400002b7          	lui	t0,0x40000
800001ea:	f0028053          	fmv.w.x	ft0,t0
800001ee:	28050553          	fmin.s	fa0,fa0,ft0
800001f2:	c00002b7          	lui	t0,0xc0000
800001f6:	f0028053          	fmv.w.x	ft0,t0
800001fa:	28051553          	fmax.s	fa0,fa0,ft0
800001fe:	3f8002b7          	lui	t0,0x3f800
80000202:	f0028053          	fmv.w.x	ft0,t0
80000206:	00a07053          	fadd.s	ft0,ft0,fa0
8000020a:	10a570d3          	fmul.s	ft1,fa0,fa0
8000020e:	3f0002b7          	lui	t0,0x3f000
80000212:	f0028153          	fmv.w.x	ft2,t0
80000216:	1020f0d3          	fmul.s	ft1,ft1,ft2
8000021a:	00107053          	fadd.s	ft0,ft0,ft1
8000021e:	10a0f0d3          	fmul.s	ft1,ft1,fa0
80000222:	3e2ab2b7          	lui	t0,0x3e2ab
80000226:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
8000022a:	f0028153          	fmv.w.x	ft2,t0
8000022e:	1020f0d3          	fmul.s	ft1,ft1,ft2
80000232:	00107553          	fadd.s	fa0,ft0,ft1
80000236:	8082                	c.jr	ra

80000238 <sin_f>:
80000238:	20a50053          	fsgnj.s	ft0,fa0,fa0
8000023c:	10a570d3          	fmul.s	ft1,fa0,fa0
80000240:	10a0f153          	fmul.s	ft2,ft1,fa0
80000244:	3e2ab2b7          	lui	t0,0x3e2ab
80000248:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
8000024c:	f00281d3          	fmv.w.x	ft3,t0
80000250:	10317153          	fmul.s	ft2,ft2,ft3
80000254:	08207553          	fsub.s	fa0,ft0,ft2
80000258:	8082                	c.jr	ra

8000025a <cos_f>:
8000025a:	3f8002b7          	lui	t0,0x3f800
8000025e:	f0028053          	fmv.w.x	ft0,t0
80000262:	10a570d3          	fmul.s	ft1,fa0,fa0
80000266:	3f0002b7          	lui	t0,0x3f000
8000026a:	f0028153          	fmv.w.x	ft2,t0
8000026e:	1020f0d3          	fmul.s	ft1,ft1,ft2
80000272:	08107553          	fsub.s	fa0,ft0,ft1
80000276:	8082                	c.jr	ra

80000278 <complex_mul>:
80000278:	10c57053          	fmul.s	ft0,fa0,fa2
8000027c:	10d5f0d3          	fmul.s	ft1,fa1,fa3
80000280:	08107153          	fsub.s	ft2,ft0,ft1
80000284:	10d57053          	fmul.s	ft0,fa0,fa3
80000288:	10c5f0d3          	fmul.s	ft1,fa1,fa2
8000028c:	001071d3          	fadd.s	ft3,ft0,ft1
80000290:	20210553          	fsgnj.s	fa0,ft2,ft2
80000294:	203185d3          	fsgnj.s	fa1,ft3,ft3
80000298:	8082                	c.jr	ra

8000029a <tanh_f>:
8000029a:	1141                	c.addi	sp,-16
8000029c:	c606                	c.swsp	ra,12(sp)
8000029e:	00a57553          	fadd.s	fa0,fa0,fa0
800002a2:	3791                	c.jal	800001e6 <exp_f>
800002a4:	3f8002b7          	lui	t0,0x3f800
800002a8:	f0028053          	fmv.w.x	ft0,t0
800002ac:	080575d3          	fsub.s	fa1,fa0,ft0
800002b0:	00057653          	fadd.s	fa2,fa0,ft0
800002b4:	18c5f553          	fdiv.s	fa0,fa1,fa2
800002b8:	40b2                	c.lwsp	ra,12(sp)
800002ba:	0141                	c.addi	sp,16
800002bc:	8082                	c.jr	ra
