
build/exe/hilbert_scan.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <_start>:
80000000:	70040117          	auipc	sp,0x70040
80000004:	00010113          	addi	sp,sp,0 # f0040000 <_stack_start>
80000008:	80001537          	lui	a0,0x80001
8000000c:	800025b7          	lui	a1,0x80002
80000010:	460d                	c.li	a2,3
80000012:	2039                	c.jal	80000020 <hilbert_scan>
80000014:	d05802b7          	lui	t0,0xd0580
80000018:	4305                	c.li	t1,1
8000001a:	0062a023          	sw	t1,0(t0) # d0580000 <_end+0x5057feec>
8000001e:	a001                	c.j	8000001e <_start+0x1e>

80000020 <hilbert_scan>:
80000020:	7179                	c.addi16sp	sp,-48
80000022:	d606                	c.swsp	ra,44(sp)
80000024:	d422                	c.swsp	s0,40(sp)
80000026:	d226                	c.swsp	s1,36(sp)
80000028:	d04a                	c.swsp	s2,32(sp)
8000002a:	ce4e                	c.swsp	s3,28(sp)
8000002c:	cc52                	c.swsp	s4,24(sp)
8000002e:	ca56                	c.swsp	s5,20(sp)
80000030:	c85a                	c.swsp	s6,16(sp)
80000032:	842a                	c.mv	s0,a0
80000034:	84ae                	c.mv	s1,a1
80000036:	8932                	c.mv	s2,a2
80000038:	c839                	c.beqz	s0,8000008e <error>
8000003a:	c8b1                	c.beqz	s1,8000008e <error>
8000003c:	04090963          	beq	s2,zero,8000008e <error>
80000040:	4981                	c.li	s3,0

80000042 <pixel_loop>:
80000042:	6285                	c.lui	t0,0x1
80000044:	0459d363          	bge	s3,t0,8000008a <success>
80000048:	04000513          	addi	a0,zero,64
8000004c:	85ce                	c.mv	a1,s3
8000004e:	2899                	c.jal	800000a4 <d2xy>
80000050:	8aaa                	c.mv	s5,a0
80000052:	8b2e                	c.mv	s6,a1
80000054:	4a01                	c.li	s4,0

80000056 <channel_loop>:
80000056:	032a5863          	bge	s4,s2,80000086 <next_pixel>
8000005a:	6285                	c.lui	t0,0x1
8000005c:	025a0333          	mul	t1,s4,t0
80000060:	04000293          	addi	t0,zero,64
80000064:	025a83b3          	mul	t2,s5,t0
80000068:	931e                	c.add	t1,t2
8000006a:	935a                	c.add	t1,s6
8000006c:	030a                	c.slli	t1,0x2
8000006e:	9322                	c.add	t1,s0
80000070:	00032e03          	lw	t3,0(t1)
80000074:	03298eb3          	mul	t4,s3,s2
80000078:	9ed2                	c.add	t4,s4
8000007a:	0e8a                	c.slli	t4,0x2
8000007c:	9ea6                	c.add	t4,s1
8000007e:	01cea023          	sw	t3,0(t4)
80000082:	0a05                	c.addi	s4,1
80000084:	bfc9                	c.j	80000056 <channel_loop>

80000086 <next_pixel>:
80000086:	0985                	c.addi	s3,1
80000088:	bf6d                	c.j	80000042 <pixel_loop>

8000008a <success>:
8000008a:	4501                	c.li	a0,0
8000008c:	a011                	c.j	80000090 <done>

8000008e <error>:
8000008e:	557d                	c.li	a0,-1

80000090 <done>:
80000090:	50b2                	c.lwsp	ra,44(sp)
80000092:	5422                	c.lwsp	s0,40(sp)
80000094:	5492                	c.lwsp	s1,36(sp)
80000096:	5902                	c.lwsp	s2,32(sp)
80000098:	49f2                	c.lwsp	s3,28(sp)
8000009a:	4a62                	c.lwsp	s4,24(sp)
8000009c:	4ad2                	c.lwsp	s5,20(sp)
8000009e:	4b42                	c.lwsp	s6,16(sp)
800000a0:	6145                	c.addi16sp	sp,48
800000a2:	8082                	c.jr	ra

800000a4 <d2xy>:
800000a4:	1101                	c.addi	sp,-32
800000a6:	ce22                	c.swsp	s0,28(sp)
800000a8:	cc26                	c.swsp	s1,24(sp)
800000aa:	ca4a                	c.swsp	s2,20(sp)
800000ac:	c84e                	c.swsp	s3,16(sp)
800000ae:	c652                	c.swsp	s4,12(sp)
800000b0:	c456                	c.swsp	s5,8(sp)
800000b2:	c25a                	c.swsp	s6,4(sp)
800000b4:	842a                	c.mv	s0,a0
800000b6:	84ae                	c.mv	s1,a1
800000b8:	4901                	c.li	s2,0
800000ba:	4981                	c.li	s3,0
800000bc:	4a05                	c.li	s4,1

800000be <d2xy_loop>:
800000be:	048a5063          	bge	s4,s0,800000fe <d2xy_done>
800000c2:	0014da93          	srli	s5,s1,0x1
800000c6:	001afa93          	andi	s5,s5,1
800000ca:	0154cb33          	xor	s6,s1,s5
800000ce:	001b7b13          	andi	s6,s6,1
800000d2:	000b1d63          	bne	s6,zero,800000ec <d2xy_skip_rot>
800000d6:	000a8863          	beq	s5,zero,800000e6 <d2xy_skip_flip>
800000da:	fffa0293          	addi	t0,s4,-1
800000de:	41228933          	sub	s2,t0,s2
800000e2:	413289b3          	sub	s3,t0,s3

800000e6 <d2xy_skip_flip>:
800000e6:	82ca                	c.mv	t0,s2
800000e8:	894e                	c.mv	s2,s3
800000ea:	8996                	c.mv	s3,t0

800000ec <d2xy_skip_rot>:
800000ec:	035a02b3          	mul	t0,s4,s5
800000f0:	9916                	c.add	s2,t0
800000f2:	036a02b3          	mul	t0,s4,s6
800000f6:	9996                	c.add	s3,t0
800000f8:	8089                	c.srli	s1,0x2
800000fa:	0a06                	c.slli	s4,0x1
800000fc:	b7c9                	c.j	800000be <d2xy_loop>

800000fe <d2xy_done>:
800000fe:	854a                	c.mv	a0,s2
80000100:	85ce                	c.mv	a1,s3
80000102:	4472                	c.lwsp	s0,28(sp)
80000104:	44e2                	c.lwsp	s1,24(sp)
80000106:	4952                	c.lwsp	s2,20(sp)
80000108:	49c2                	c.lwsp	s3,16(sp)
8000010a:	4a32                	c.lwsp	s4,12(sp)
8000010c:	4aa2                	c.lwsp	s5,8(sp)
8000010e:	4b12                	c.lwsp	s6,4(sp)
80000110:	6105                	c.addi16sp	sp,32
80000112:	8082                	c.jr	ra
