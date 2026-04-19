
build/exe/hilbert_scan.exe:     file format elf32-littleriscv


Disassembly of section .text:

80000000 <hilbert_scan>:
80000000:	1101                	c.addi	sp,-32
80000002:	c022                	c.swsp	s0,0(sp)
80000004:	c226                	c.swsp	s1,4(sp)
80000006:	c44a                	c.swsp	s2,8(sp)
80000008:	c64e                	c.swsp	s3,12(sp)
8000000a:	c852                	c.swsp	s4,16(sp)
8000000c:	ca56                	c.swsp	s5,20(sp)
8000000e:	cc5a                	c.swsp	s6,24(sp)
80000010:	ce5e                	c.swsp	s7,28(sp)
80000012:	842a                	c.mv	s0,a0
80000014:	84ae                	c.mv	s1,a1
80000016:	8932                	c.mv	s2,a2
80000018:	89b6                	c.mv	s3,a3
8000001a:	4a01                	c.li	s4,0
8000001c:	6b85                	c.lui	s7,0x1
8000001e:	6285                	c.lui	t0,0x1

80000020 <outer_loop>:
80000020:	037a5c63          	bge	s4,s7,80000058 <done>
80000024:	002a1313          	slli	t1,s4,0x2
80000028:	9322                	c.add	t1,s0
8000002a:	00032b03          	lw	s6,0(t1)
8000002e:	4a81                	c.li	s5,0

80000030 <inner_loop>:
80000030:	033ad263          	bge	s5,s3,80000054 <inner_done>
80000034:	025a83b3          	mul	t2,s5,t0
80000038:	93da                	c.add	t2,s6
8000003a:	038a                	c.slli	t2,0x2
8000003c:	93a6                	c.add	t2,s1
8000003e:	0003ae03          	lw	t3,0(t2)
80000042:	033a0eb3          	mul	t4,s4,s3
80000046:	9ed6                	c.add	t4,s5
80000048:	0e8a                	c.slli	t4,0x2
8000004a:	9eca                	c.add	t4,s2
8000004c:	01cea023          	sw	t3,0(t4)
80000050:	0a85                	c.addi	s5,1
80000052:	bff9                	c.j	80000030 <inner_loop>

80000054 <inner_done>:
80000054:	0a05                	c.addi	s4,1
80000056:	b7e9                	c.j	80000020 <outer_loop>

80000058 <done>:
80000058:	4402                	c.lwsp	s0,0(sp)
8000005a:	4492                	c.lwsp	s1,4(sp)
8000005c:	4922                	c.lwsp	s2,8(sp)
8000005e:	49b2                	c.lwsp	s3,12(sp)
80000060:	4a42                	c.lwsp	s4,16(sp)
80000062:	4ad2                	c.lwsp	s5,20(sp)
80000064:	4b62                	c.lwsp	s6,24(sp)
80000066:	4bf2                	c.lwsp	s7,28(sp)
80000068:	6105                	c.addi16sp	sp,32
8000006a:	8082                	c.jr	ra
8000006c:	0001                	c.addi	zero,0
