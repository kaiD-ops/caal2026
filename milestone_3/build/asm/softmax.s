
build/exe/softmax.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70040117          	auipc	sp,0x70040
80000004:	01010113          	addi	sp,sp,16 # f0040010 <_stack_start>
80000008:	70040517          	auipc	a0,0x70040
8000000c:	ff850513          	addi	a0,a0,-8 # f0040000 <test_data>
80000010:	20a5                	c.jal	80000078 <softmax>
80000012:	d05802b7          	lui	t0,0xd0580
80000016:	4305                	c.li	t1,1
80000018:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057feee>
8000001c:	a001                	c.j	8000001c <_start+0x1c>

8000001e <exp_f>:
8000001e:	410002b7          	lui	t0,0x41000
80000022:	f0028053          	fmv.w.x	ft0,t0
80000026:	a00512d3          	flt.s	t0,fa0,ft0
8000002a:	02029d63          	bne	t0,zero,80000064 <clamp_neg>
8000002e:	c10002b7          	lui	t0,0xc1000
80000032:	f0028053          	fmv.w.x	ft0,t0
80000036:	a0a012d3          	flt.s	t0,ft0,fa0
8000003a:	02029863          	bne	t0,zero,8000006a <clamp_pos>
8000003e:	3fb8b2b7          	lui	t0,0x3fb8b
80000042:	a3b28293          	addi	t0,t0,-1477 # 3fb8aa3b <_start-0x404755c5>
80000046:	f0028053          	fmv.w.x	ft0,t0
8000004a:	10057553          	fmul.s	fa0,fa0,ft0
8000004e:	c00512d3          	fcvt.w.s	t0,fa0,rtz
80000052:	07f28293          	addi	t0,t0,127
80000056:	02de                	c.slli	t0,0x17
80000058:	3f800337          	lui	t1,0x3f800
8000005c:	929a                	c.add	t0,t1
8000005e:	f0028553          	fmv.w.x	fa0,t0
80000062:	8082                	c.jr	ra

80000064 <clamp_neg>:
80000064:	f0000553          	fmv.w.x	fa0,zero
80000068:	8082                	c.jr	ra

8000006a <clamp_pos>:
8000006a:	461c42b7          	lui	t0,0x461c4
8000006e:	c0028293          	addi	t0,t0,-1024 # 461c3c00 <_start-0x39e3c400>
80000072:	f0028553          	fmv.w.x	fa0,t0
80000076:	8082                	c.jr	ra

80000078 <softmax>:
80000078:	7179                	c.addi16sp	sp,-48
8000007a:	d606                	c.swsp	ra,44(sp)
8000007c:	d422                	c.swsp	s0,40(sp)
8000007e:	f222                	c.fswsp	fs0,36(sp)
80000080:	f026                	c.fswsp	fs1,32(sp)
80000082:	ee4a                	c.fswsp	fs2,28(sp)
80000084:	ec4e                	c.fswsp	fs3,24(sp)
80000086:	842a                	c.mv	s0,a0
80000088:	cc25                	c.beqz	s0,80000100 <error>
8000008a:	6000                	c.flw	fs0,0(s0)
8000008c:	00442007          	flw	ft0,4(s0)
80000090:	28041453          	fmax.s	fs0,fs0,ft0
80000094:	00842007          	flw	ft0,8(s0)
80000098:	28041453          	fmax.s	fs0,fs0,ft0
8000009c:	00c42007          	flw	ft0,12(s0)
800000a0:	28041453          	fmax.s	fs0,fs0,ft0
800000a4:	6008                	c.flw	fa0,0(s0)
800000a6:	08857553          	fsub.s	fa0,fa0,fs0
800000aa:	3f95                	c.jal	8000001e <exp_f>
800000ac:	e008                	c.fsw	fa0,0(s0)
800000ae:	20a504d3          	fsgnj.s	fs1,fa0,fa0
800000b2:	6048                	c.flw	fa0,4(s0)
800000b4:	08857553          	fsub.s	fa0,fa0,fs0
800000b8:	379d                	c.jal	8000001e <exp_f>
800000ba:	e048                	c.fsw	fa0,4(s0)
800000bc:	00a4f4d3          	fadd.s	fs1,fs1,fa0
800000c0:	6408                	c.flw	fa0,8(s0)
800000c2:	08857553          	fsub.s	fa0,fa0,fs0
800000c6:	3fa1                	c.jal	8000001e <exp_f>
800000c8:	e408                	c.fsw	fa0,8(s0)
800000ca:	00a4f4d3          	fadd.s	fs1,fs1,fa0
800000ce:	6448                	c.flw	fa0,12(s0)
800000d0:	08857553          	fsub.s	fa0,fa0,fs0
800000d4:	37a9                	c.jal	8000001e <exp_f>
800000d6:	e448                	c.fsw	fa0,12(s0)
800000d8:	00a4f4d3          	fadd.s	fs1,fs1,fa0
800000dc:	6008                	c.flw	fa0,0(s0)
800000de:	18957553          	fdiv.s	fa0,fa0,fs1
800000e2:	e008                	c.fsw	fa0,0(s0)
800000e4:	6048                	c.flw	fa0,4(s0)
800000e6:	18957553          	fdiv.s	fa0,fa0,fs1
800000ea:	e048                	c.fsw	fa0,4(s0)
800000ec:	6408                	c.flw	fa0,8(s0)
800000ee:	18957553          	fdiv.s	fa0,fa0,fs1
800000f2:	e408                	c.fsw	fa0,8(s0)
800000f4:	6448                	c.flw	fa0,12(s0)
800000f6:	18957553          	fdiv.s	fa0,fa0,fs1
800000fa:	e448                	c.fsw	fa0,12(s0)
800000fc:	4501                	c.li	a0,0
800000fe:	a011                	c.j	80000102 <done>

80000100 <error>:
80000100:	557d                	c.li	a0,-1

80000102 <done>:
80000102:	7412                	c.flwsp	fs0,36(sp)
80000104:	7482                	c.flwsp	fs1,32(sp)
80000106:	6972                	c.flwsp	fs2,28(sp)
80000108:	69e2                	c.flwsp	fs3,24(sp)
8000010a:	50b2                	c.lwsp	ra,44(sp)
8000010c:	5422                	c.lwsp	s0,40(sp)
8000010e:	6145                	c.addi16sp	sp,48
80000110:	8082                	c.jr	ra
