
build/exe/linear_layer.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70040117          	auipc	sp,0x70040
80000004:	16010113          	addi	sp,sp,352 # f0040160 <_stack_start>
80000008:	70040517          	auipc	a0,0x70040
8000000c:	ff850513          	addi	a0,a0,-8 # f0040000 <test_input>
80000010:	70040597          	auipc	a1,0x70040
80000014:	01058593          	addi	a1,a1,16 # f0040020 <test_weight>
80000018:	70040617          	auipc	a2,0x70040
8000001c:	03860613          	addi	a2,a2,56 # f0040050 <test_bias>
80000020:	70040697          	auipc	a3,0x70040
80000024:	04068693          	addi	a3,a3,64 # f0040060 <test_output>
80000028:	4711                	c.li	a4,4
8000002a:	478d                	c.li	a5,3
8000002c:	4809                	c.li	a6,2
8000002e:	2039                	c.jal	8000003c <linear_layer>
80000030:	d05802b7          	lui	t0,0xd0580
80000034:	4305                	c.li	t1,1
80000036:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057fef2>
8000003a:	a001                	c.j	8000003a <_start+0x3a>

8000003c <linear_layer>:
8000003c:	715d                	c.addi16sp	sp,-80
8000003e:	c686                	c.swsp	ra,76(sp)
80000040:	c4a2                	c.swsp	s0,72(sp)
80000042:	c2a6                	c.swsp	s1,68(sp)
80000044:	c0ca                	c.swsp	s2,64(sp)
80000046:	de4e                	c.swsp	s3,60(sp)
80000048:	dc52                	c.swsp	s4,56(sp)
8000004a:	da56                	c.swsp	s5,52(sp)
8000004c:	d85a                	c.swsp	s6,48(sp)
8000004e:	d65e                	c.swsp	s7,44(sp)
80000050:	d462                	c.swsp	s8,40(sp)
80000052:	d266                	c.swsp	s9,36(sp)
80000054:	d06a                	c.swsp	s10,32(sp)
80000056:	ce6e                	c.swsp	s11,28(sp)
80000058:	ec02                	c.fswsp	ft0,24(sp)
8000005a:	ea06                	c.fswsp	ft1,20(sp)
8000005c:	e80a                	c.fswsp	ft2,16(sp)
8000005e:	e60e                	c.fswsp	ft3,12(sp)
80000060:	842a                	c.mv	s0,a0
80000062:	84ae                	c.mv	s1,a1
80000064:	8932                	c.mv	s2,a2
80000066:	89b6                	c.mv	s3,a3
80000068:	8a3a                	c.mv	s4,a4
8000006a:	8abe                	c.mv	s5,a5
8000006c:	8b42                	c.mv	s6,a6
8000006e:	c83d                	c.beqz	s0,800000e4 <error>
80000070:	c8b5                	c.beqz	s1,800000e4 <error>
80000072:	06090963          	beq	s2,zero,800000e4 <error>
80000076:	06098763          	beq	s3,zero,800000e4 <error>
8000007a:	060a0563          	beq	s4,zero,800000e4 <error>
8000007e:	060a8363          	beq	s5,zero,800000e4 <error>
80000082:	060b0163          	beq	s6,zero,800000e4 <error>
80000086:	4b81                	c.li	s7,0

80000088 <row_loop>:
80000088:	076bd063          	bge	s7,s6,800000e8 <row_done>
8000008c:	4c01                	c.li	s8,0

8000008e <col_loop>:
8000008e:	055c5963          	bge	s8,s5,800000e0 <col_done>
80000092:	002c1293          	slli	t0,s8,0x2
80000096:	92ca                	c.add	t0,s2
80000098:	0002a007          	flw	ft0,0(t0)
8000009c:	4c81                	c.li	s9,0
8000009e:	4d01                	c.li	s10,0
800000a0:	4d81                	c.li	s11,0

800000a2 <inner_loop>:
800000a2:	034cd663          	bge	s9,s4,800000ce <inner_done>
800000a6:	034b8333          	mul	t1,s7,s4
800000aa:	9366                	c.add	t1,s9
800000ac:	030a                	c.slli	t1,0x2
800000ae:	9322                	c.add	t1,s0
800000b0:	00032087          	flw	ft1,0(t1)
800000b4:	035c83b3          	mul	t2,s9,s5
800000b8:	93e2                	c.add	t2,s8
800000ba:	038a                	c.slli	t2,0x2
800000bc:	93a6                	c.add	t2,s1
800000be:	0003a107          	flw	ft2,0(t2)
800000c2:	1020f1d3          	fmul.s	ft3,ft1,ft2
800000c6:	00307053          	fadd.s	ft0,ft0,ft3
800000ca:	0c85                	c.addi	s9,1
800000cc:	bfd9                	c.j	800000a2 <inner_loop>

800000ce <inner_done>:
800000ce:	035b8e33          	mul	t3,s7,s5
800000d2:	9e62                	c.add	t3,s8
800000d4:	0e0a                	c.slli	t3,0x2
800000d6:	9e4e                	c.add	t3,s3
800000d8:	000e2027          	fsw	ft0,0(t3)
800000dc:	0c05                	c.addi	s8,1
800000de:	bf45                	c.j	8000008e <col_loop>

800000e0 <col_done>:
800000e0:	0b85                	c.addi	s7,1
800000e2:	b75d                	c.j	80000088 <row_loop>

800000e4 <error>:
800000e4:	557d                	c.li	a0,-1
800000e6:	a009                	c.j	800000e8 <row_done>

800000e8 <row_done>:
800000e8:	40b6                	c.lwsp	ra,76(sp)
800000ea:	4426                	c.lwsp	s0,72(sp)
800000ec:	4496                	c.lwsp	s1,68(sp)
800000ee:	4906                	c.lwsp	s2,64(sp)
800000f0:	59f2                	c.lwsp	s3,60(sp)
800000f2:	5a62                	c.lwsp	s4,56(sp)
800000f4:	5ad2                	c.lwsp	s5,52(sp)
800000f6:	5b42                	c.lwsp	s6,48(sp)
800000f8:	5bb2                	c.lwsp	s7,44(sp)
800000fa:	5c22                	c.lwsp	s8,40(sp)
800000fc:	5c92                	c.lwsp	s9,36(sp)
800000fe:	5d02                	c.lwsp	s10,32(sp)
80000100:	4df2                	c.lwsp	s11,28(sp)
80000102:	6062                	c.flwsp	ft0,24(sp)
80000104:	60d2                	c.flwsp	ft1,20(sp)
80000106:	6142                	c.flwsp	ft2,16(sp)
80000108:	61b2                	c.flwsp	ft3,12(sp)
8000010a:	6161                	c.addi16sp	sp,80
8000010c:	8082                	c.jr	ra
