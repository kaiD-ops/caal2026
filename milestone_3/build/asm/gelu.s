
build/exe/gelu.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70040117          	auipc	sp,0x70040
80000004:	03010113          	addi	sp,sp,48 # f0040030 <_stack_start>
80000008:	70040517          	auipc	a0,0x70040
8000000c:	ff850513          	addi	a0,a0,-8 # f0040000 <test_data>
80000010:	4591                	c.li	a1,4
80000012:	2039                	c.jal	80000020 <gelu_vec>
80000014:	d05802b7          	lui	t0,0xd0580
80000018:	4305                	c.li	t1,1
8000001a:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057fe66>
8000001e:	a001                	c.j	8000001e <_start+0x1e>

80000020 <gelu_vec>:
80000020:	7179                	c.addi16sp	sp,-48
80000022:	d606                	c.swsp	ra,44(sp)
80000024:	d422                	c.swsp	s0,40(sp)
80000026:	d226                	c.swsp	s1,36(sp)
80000028:	d04a                	c.swsp	s2,32(sp)
8000002a:	ee22                	c.fswsp	fs0,28(sp)
8000002c:	ec26                	c.fswsp	fs1,24(sp)
8000002e:	ea4a                	c.fswsp	fs2,20(sp)
80000030:	e84e                	c.fswsp	fs3,16(sp)
80000032:	842a                	c.mv	s0,a0
80000034:	84ae                	c.mv	s1,a1
80000036:	4901                	c.li	s2,0
80000038:	f00402b7          	lui	t0,0xf0040
8000003c:	0142a007          	flw	ft0,20(t0) # f0040014 <gelu_c1>
80000040:	200004d3          	fsgnj.s	fs1,ft0,ft0
80000044:	f00402b7          	lui	t0,0xf0040
80000048:	0182a007          	flw	ft0,24(t0) # f0040018 <gelu_c2>
8000004c:	20000953          	fsgnj.s	fs2,ft0,ft0
80000050:	f00402b7          	lui	t0,0xf0040
80000054:	01c2a007          	flw	ft0,28(t0) # f004001c <one_f_g>
80000058:	200009d3          	fsgnj.s	fs3,ft0,ft0
8000005c:	f00402b7          	lui	t0,0xf0040
80000060:	0202a007          	flw	ft0,32(t0) # f0040020 <half_f>
80000064:	20000053          	fsgnj.s	ft0,ft0,ft0

80000068 <gelu_loop>:
80000068:	04995363          	bge	s2,s1,800000ae <gelu_done>
8000006c:	00291293          	slli	t0,s2,0x2
80000070:	00540333          	add	t1,s0,t0
80000074:	00032407          	flw	fs0,0(t1)
80000078:	108470d3          	fmul.s	ft1,fs0,fs0
8000007c:	1080f0d3          	fmul.s	ft1,ft1,fs0
80000080:	1014f0d3          	fmul.s	ft1,fs1,ft1
80000084:	001470d3          	fadd.s	ft1,fs0,ft1
80000088:	10197553          	fmul.s	fa0,fs2,ft1
8000008c:	20ed                	c.jal	80000176 <tanh_f>
8000008e:	01357553          	fadd.s	fa0,fa0,fs3
80000092:	10857553          	fmul.s	fa0,fa0,fs0
80000096:	f00403b7          	lui	t2,0xf0040
8000009a:	0203a087          	flw	ft1,32(t2) # f0040020 <half_f>
8000009e:	10157553          	fmul.s	fa0,fa0,ft1
800000a2:	00540333          	add	t1,s0,t0
800000a6:	00a32027          	fsw	fa0,0(t1)
800000aa:	0905                	c.addi	s2,1
800000ac:	bf75                	c.j	80000068 <gelu_loop>

800000ae <gelu_done>:
800000ae:	50b2                	c.lwsp	ra,44(sp)
800000b0:	5422                	c.lwsp	s0,40(sp)
800000b2:	5492                	c.lwsp	s1,36(sp)
800000b4:	5902                	c.lwsp	s2,32(sp)
800000b6:	6472                	c.flwsp	fs0,28(sp)
800000b8:	64e2                	c.flwsp	fs1,24(sp)
800000ba:	6952                	c.flwsp	fs2,20(sp)
800000bc:	69c2                	c.flwsp	fs3,16(sp)
800000be:	6145                	c.addi16sp	sp,48
800000c0:	8082                	c.jr	ra

800000c2 <exp_f>:
800000c2:	400002b7          	lui	t0,0x40000
800000c6:	f0028053          	fmv.w.x	ft0,t0
800000ca:	28050553          	fmin.s	fa0,fa0,ft0
800000ce:	c00002b7          	lui	t0,0xc0000
800000d2:	f0028053          	fmv.w.x	ft0,t0
800000d6:	28051553          	fmax.s	fa0,fa0,ft0
800000da:	3f8002b7          	lui	t0,0x3f800
800000de:	f0028053          	fmv.w.x	ft0,t0
800000e2:	00a07053          	fadd.s	ft0,ft0,fa0
800000e6:	10a570d3          	fmul.s	ft1,fa0,fa0
800000ea:	3f0002b7          	lui	t0,0x3f000
800000ee:	f0028153          	fmv.w.x	ft2,t0
800000f2:	1020f0d3          	fmul.s	ft1,ft1,ft2
800000f6:	00107053          	fadd.s	ft0,ft0,ft1
800000fa:	10a0f0d3          	fmul.s	ft1,ft1,fa0
800000fe:	3e2ab2b7          	lui	t0,0x3e2ab
80000102:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
80000106:	f0028153          	fmv.w.x	ft2,t0
8000010a:	1020f0d3          	fmul.s	ft1,ft1,ft2
8000010e:	00107553          	fadd.s	fa0,ft0,ft1
80000112:	8082                	c.jr	ra

80000114 <sin_f>:
80000114:	20a50053          	fsgnj.s	ft0,fa0,fa0
80000118:	10a570d3          	fmul.s	ft1,fa0,fa0
8000011c:	10a0f153          	fmul.s	ft2,ft1,fa0
80000120:	3e2ab2b7          	lui	t0,0x3e2ab
80000124:	aab28293          	addi	t0,t0,-1365 # 3e2aaaab <_start-0x41d55555>
80000128:	f00281d3          	fmv.w.x	ft3,t0
8000012c:	10317153          	fmul.s	ft2,ft2,ft3
80000130:	08207553          	fsub.s	fa0,ft0,ft2
80000134:	8082                	c.jr	ra

80000136 <cos_f>:
80000136:	3f8002b7          	lui	t0,0x3f800
8000013a:	f0028053          	fmv.w.x	ft0,t0
8000013e:	10a570d3          	fmul.s	ft1,fa0,fa0
80000142:	3f0002b7          	lui	t0,0x3f000
80000146:	f0028153          	fmv.w.x	ft2,t0
8000014a:	1020f0d3          	fmul.s	ft1,ft1,ft2
8000014e:	08107553          	fsub.s	fa0,ft0,ft1
80000152:	8082                	c.jr	ra

80000154 <complex_mul>:
80000154:	10c57053          	fmul.s	ft0,fa0,fa2
80000158:	10d5f0d3          	fmul.s	ft1,fa1,fa3
8000015c:	08107153          	fsub.s	ft2,ft0,ft1
80000160:	10d57053          	fmul.s	ft0,fa0,fa3
80000164:	10c5f0d3          	fmul.s	ft1,fa1,fa2
80000168:	001071d3          	fadd.s	ft3,ft0,ft1
8000016c:	20210553          	fsgnj.s	fa0,ft2,ft2
80000170:	203185d3          	fsgnj.s	fa1,ft3,ft3
80000174:	8082                	c.jr	ra

80000176 <tanh_f>:
80000176:	1141                	c.addi	sp,-16
80000178:	c606                	c.swsp	ra,12(sp)
8000017a:	00a57553          	fadd.s	fa0,fa0,fa0
8000017e:	3791                	c.jal	800000c2 <exp_f>
80000180:	3f8002b7          	lui	t0,0x3f800
80000184:	f0028053          	fmv.w.x	ft0,t0
80000188:	080575d3          	fsub.s	fa1,fa0,ft0
8000018c:	00057653          	fadd.s	fa2,fa0,ft0
80000190:	18c5f553          	fdiv.s	fa0,fa1,fa2
80000194:	40b2                	c.lwsp	ra,12(sp)
80000196:	0141                	c.addi	sp,16
80000198:	8082                	c.jr	ra
