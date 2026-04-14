	.file	"nn.c"
	.text
	.local	buf_hilbert
	.comm	buf_hilbert,16384,32
	.local	buf_proj
	.comm	buf_proj,1048576,32
	.local	buf_s4d1
	.comm	buf_s4d1,1048576,32
	.local	buf_s4d2
	.comm	buf_s4d2,1048576,32
	.local	buf_pooled
	.comm	buf_pooled,256,32
	.local	buf_logits
	.comm	buf_logits,16,16
	.local	s4d_kernel
	.comm	s4d_kernel,16384,32
	.section	.rodata
.LC0:
	.string	"rb"
.LC1:
	.string	"[ERROR] Cannot open: %s\n"
.LC2:
	.string	"[ERROR] Unexpected EOF in %s\n"
	.text
	.globl	load_weights
	.type	load_weights, @function
load_weights:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	-24(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	cmpq	$0, -16(%rbp)
	jne	.L2
	movq	stderr(%rip), %rax
	movq	-24(%rbp), %rdx
	leaq	.LC1(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$-1, %eax
	jmp	.L3
.L2:
	movq	-32(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, %rcx
	movl	$4096, %edx
	movl	$4, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$4096, -8(%rbp)
	jne	.L20
	movq	-32(%rbp), %rax
	leaq	16384(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$64, -8(%rbp)
	jne	.L21
	movq	-32(%rbp), %rax
	leaq	16640(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$64, -8(%rbp)
	jne	.L22
	movq	-32(%rbp), %rax
	leaq	16896(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$64, -8(%rbp)
	jne	.L23
	movq	-32(%rbp), %rax
	leaq	17152(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$2048, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$2048, -8(%rbp)
	jne	.L24
	movq	-32(%rbp), %rax
	leaq	25344(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$2048, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$2048, -8(%rbp)
	jne	.L25
	movq	-32(%rbp), %rax
	leaq	33536(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$4096, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$4096, -8(%rbp)
	jne	.L26
	movq	-32(%rbp), %rax
	leaq	49920(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$64, -8(%rbp)
	jne	.L27
	movq	-32(%rbp), %rax
	leaq	50176(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$64, -8(%rbp)
	jne	.L28
	movq	-32(%rbp), %rax
	leaq	50432(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$2048, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$2048, -8(%rbp)
	jne	.L29
	movq	-32(%rbp), %rax
	leaq	58624(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$2048, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$2048, -8(%rbp)
	jne	.L30
	movq	-32(%rbp), %rax
	leaq	66816(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$4096, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$4096, -8(%rbp)
	jne	.L31
	movq	-32(%rbp), %rax
	leaq	83200(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$64, -8(%rbp)
	jne	.L32
	movq	-32(%rbp), %rax
	leaq	83456(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$256, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$256, -8(%rbp)
	jne	.L33
	movq	-32(%rbp), %rax
	leaq	84480(%rax), %rdi
	movq	-16(%rbp), %rax
	movq	%rax, %rcx
	movl	$4, %edx
	movl	$4, %esi
	call	fread@PLT
	movq	%rax, -8(%rbp)
	cmpq	$4, -8(%rbp)
	jne	.L34
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	$0, %eax
	jmp	.L3
.L20:
	nop
	jmp	.L5
.L21:
	nop
	jmp	.L5
.L22:
	nop
	jmp	.L5
.L23:
	nop
	jmp	.L5
.L24:
	nop
	jmp	.L5
.L25:
	nop
	jmp	.L5
.L26:
	nop
	jmp	.L5
.L27:
	nop
	jmp	.L5
.L28:
	nop
	jmp	.L5
.L29:
	nop
	jmp	.L5
.L30:
	nop
	jmp	.L5
.L31:
	nop
	jmp	.L5
.L32:
	nop
	jmp	.L5
.L33:
	nop
	jmp	.L5
.L34:
	nop
.L5:
	movq	stderr(%rip), %rax
	movq	-24(%rbp), %rdx
	leaq	.LC2(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movl	$-1, %eax
.L3:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	load_weights, .-load_weights
	.globl	hilbert_scan
	.type	hilbert_scan, @function
hilbert_scan:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movl	$0, -12(%rbp)
	jmp	.L36
.L39:
	movq	-24(%rbp), %rax
	movl	-12(%rbp), %edx
	movslq	%edx, %rdx
	movl	(%rax,%rdx,4), %eax
	movl	%eax, -4(%rbp)
	movl	$0, -8(%rbp)
	jmp	.L37
.L38:
	movl	-8(%rbp), %eax
	sall	$12, %eax
	movl	%eax, %edx
	movl	-4(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	movl	-12(%rbp), %ecx
	movl	-8(%rbp), %eax
	addl	%ecx, %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-40(%rbp), %rax
	addq	%rcx, %rax
	movss	(%rdx), %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -8(%rbp)
.L37:
	cmpl	$0, -8(%rbp)
	jle	.L38
	addl	$1, -12(%rbp)
.L36:
	cmpl	$4095, -12(%rbp)
	jle	.L39
	nop
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	hilbert_scan, .-hilbert_scan
	.globl	linear_layer
	.type	linear_layer, @function
linear_layer:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%rcx, -80(%rbp)
	movl	%r8d, -84(%rbp)
	movl	%r9d, -88(%rbp)
	movl	$0, -40(%rbp)
	jmp	.L41
.L46:
	movl	-40(%rbp), %eax
	imull	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-72(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -24(%rbp)
	movl	-40(%rbp), %eax
	imull	-88(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -16(%rbp)
	movl	$0, -36(%rbp)
	jmp	.L42
.L45:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-64(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -28(%rbp)
	movl	-36(%rbp), %eax
	imull	-84(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-56(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, -8(%rbp)
	movl	$0, -32(%rbp)
	jmp	.L43
.L44:
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-8(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	movl	-32(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	mulss	%xmm1, %xmm0
	movss	-28(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -28(%rbp)
	addl	$1, -32(%rbp)
.L43:
	movl	-32(%rbp), %eax
	cmpl	-84(%rbp), %eax
	jl	.L44
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-16(%rbp), %rax
	addq	%rdx, %rax
	movss	-28(%rbp), %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -36(%rbp)
.L42:
	movl	-36(%rbp), %eax
	cmpl	-88(%rbp), %eax
	jl	.L45
	addl	$1, -40(%rbp)
.L41:
	movl	-40(%rbp), %eax
	cmpl	16(%rbp), %eax
	jl	.L46
	nop
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	linear_layer, .-linear_layer
	.globl	s4d_layer
	.type	s4d_layer, @function
s4d_layer:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$704, %rsp
	movq	%rdi, -648(%rbp)
	movq	%rsi, -656(%rbp)
	movq	%rdx, -664(%rbp)
	movq	%rcx, -672(%rbp)
	movq	%r8, -680(%rbp)
	movq	%r9, -688(%rbp)
	movq	16(%rbp), %rax
	movq	%rax, -696(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, -636(%rbp)
	jmp	.L48
.L59:
	movl	-636(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-648(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movd	%eax, %xmm0
	call	expf@PLT
	movd	%xmm0, %eax
	movl	%eax, -612(%rbp)
	movl	$0, -632(%rbp)
	jmp	.L49
.L50:
	movl	-636(%rbp), %eax
	sall	$5, %eax
	movl	%eax, %edx
	movl	-632(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-656(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movd	%eax, %xmm0
	call	expf@PLT
	movd	%xmm0, %eax
	movss	.LC3(%rip), %xmm0
	movd	%eax, %xmm2
	xorps	%xmm0, %xmm2
	movaps	%xmm2, %xmm0
	movss	%xmm0, -576(%rbp)
	movl	-636(%rbp), %eax
	sall	$5, %eax
	movl	%eax, %edx
	movl	-632(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-664(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -572(%rbp)
	movss	-576(%rbp), %xmm0
	mulss	-612(%rbp), %xmm0
	movss	%xmm0, -568(%rbp)
	movss	-572(%rbp), %xmm0
	mulss	-612(%rbp), %xmm0
	movss	%xmm0, -564(%rbp)
	movl	-568(%rbp), %eax
	movd	%eax, %xmm0
	call	expf@PLT
	movd	%xmm0, %eax
	movl	%eax, -560(%rbp)
	movl	-564(%rbp), %eax
	movd	%eax, %xmm0
	call	cosf@PLT
	movd	%xmm0, %eax
	movd	%eax, %xmm0
	mulss	-560(%rbp), %xmm0
	movl	-632(%rbp), %eax
	cltq
	movss	%xmm0, -528(%rbp,%rax,4)
	movl	-564(%rbp), %eax
	movd	%eax, %xmm0
	call	sinf@PLT
	movd	%xmm0, %eax
	movd	%eax, %xmm0
	mulss	-560(%rbp), %xmm0
	movl	-632(%rbp), %eax
	cltq
	movss	%xmm0, -400(%rbp,%rax,4)
	movl	-632(%rbp), %eax
	cltq
	movss	-528(%rbp,%rax,4), %xmm0
	movss	.LC4(%rip), %xmm1
	subss	%xmm1, %xmm0
	movss	%xmm0, -556(%rbp)
	movl	-632(%rbp), %eax
	cltq
	movss	-400(%rbp,%rax,4), %xmm0
	movss	%xmm0, -552(%rbp)
	movl	-636(%rbp), %eax
	sall	$5, %eax
	movl	%eax, %edx
	movl	-632(%rbp), %eax
	addl	%edx, %eax
	addl	%eax, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-672(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -548(%rbp)
	movl	-636(%rbp), %eax
	sall	$5, %eax
	movl	%eax, %edx
	movl	-632(%rbp), %eax
	addl	%edx, %eax
	addl	%eax, %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-672(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -544(%rbp)
	movss	-548(%rbp), %xmm0
	mulss	-556(%rbp), %xmm0
	movss	-544(%rbp), %xmm1
	mulss	-552(%rbp), %xmm1
	subss	%xmm1, %xmm0
	movss	%xmm0, -540(%rbp)
	movss	-548(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-552(%rbp), %xmm1
	movss	-544(%rbp), %xmm0
	mulss	-556(%rbp), %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -536(%rbp)
	movss	-576(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	%xmm0, %xmm1
	movss	-572(%rbp), %xmm0
	mulss	%xmm0, %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -532(%rbp)
	movss	-540(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-576(%rbp), %xmm1
	movss	-536(%rbp), %xmm0
	mulss	-572(%rbp), %xmm0
	addss	%xmm1, %xmm0
	divss	-532(%rbp), %xmm0
	movl	-632(%rbp), %eax
	cltq
	movss	%xmm0, -272(%rbp,%rax,4)
	movss	-536(%rbp), %xmm0
	mulss	-576(%rbp), %xmm0
	movss	-540(%rbp), %xmm1
	mulss	-572(%rbp), %xmm1
	subss	%xmm1, %xmm0
	divss	-532(%rbp), %xmm0
	movl	-632(%rbp), %eax
	cltq
	movss	%xmm0, -144(%rbp,%rax,4)
	addl	$1, -632(%rbp)
.L49:
	cmpl	$31, -632(%rbp)
	jle	.L50
	movl	$0, -628(%rbp)
	jmp	.L51
.L54:
	pxor	%xmm0, %xmm0
	movss	%xmm0, -620(%rbp)
	movl	$0, -632(%rbp)
	jmp	.L52
.L53:
	movl	-636(%rbp), %eax
	sall	$5, %eax
	movl	%eax, %edx
	movl	-632(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-656(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movd	%eax, %xmm0
	call	expf@PLT
	movd	%xmm0, %eax
	movss	.LC3(%rip), %xmm0
	movd	%eax, %xmm3
	xorps	%xmm0, %xmm3
	movaps	%xmm3, %xmm0
	movss	%xmm0, -604(%rbp)
	movl	-636(%rbp), %eax
	sall	$5, %eax
	movl	%eax, %edx
	movl	-632(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-664(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -600(%rbp)
	movss	-604(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-612(%rbp), %xmm1
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-628(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -596(%rbp)
	movss	-600(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	mulss	-612(%rbp), %xmm1
	pxor	%xmm0, %xmm0
	cvtsi2ssl	-628(%rbp), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -592(%rbp)
	movl	-596(%rbp), %eax
	movd	%eax, %xmm0
	call	expf@PLT
	movd	%xmm0, %eax
	movl	%eax, -588(%rbp)
	movl	-592(%rbp), %eax
	movd	%eax, %xmm0
	call	cosf@PLT
	movss	-588(%rbp), %xmm1
	mulss	%xmm1, %xmm0
	movss	%xmm0, -584(%rbp)
	movl	-592(%rbp), %eax
	movd	%eax, %xmm0
	call	sinf@PLT
	movss	-588(%rbp), %xmm1
	mulss	%xmm1, %xmm0
	movss	%xmm0, -580(%rbp)
	movl	-632(%rbp), %eax
	cltq
	movss	-272(%rbp,%rax,4), %xmm0
	mulss	-584(%rbp), %xmm0
	movl	-632(%rbp), %eax
	cltq
	movss	-144(%rbp,%rax,4), %xmm1
	mulss	-580(%rbp), %xmm1
	subss	%xmm1, %xmm0
	addss	%xmm0, %xmm0
	movss	-620(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -620(%rbp)
	addl	$1, -632(%rbp)
.L52:
	cmpl	$31, -632(%rbp)
	jle	.L53
	movl	-628(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s4d_kernel(%rip), %rax
	movss	-620(%rbp), %xmm0
	movss	%xmm0, (%rdx,%rax)
	addl	$1, -628(%rbp)
.L51:
	cmpl	$4095, -628(%rbp)
	jle	.L54
	movl	-636(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-680(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -608(%rbp)
	movl	$0, -628(%rbp)
	jmp	.L55
.L58:
	movl	-628(%rbp), %eax
	sall	$6, %eax
	movl	%eax, %edx
	movl	-636(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-688(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	-608(%rbp), %xmm1
	mulss	%xmm1, %xmm0
	movss	%xmm0, -616(%rbp)
	movl	$0, -624(%rbp)
	jmp	.L56
.L57:
	movl	-624(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	s4d_kernel(%rip), %rax
	movss	(%rdx,%rax), %xmm1
	movl	-628(%rbp), %eax
	subl	-624(%rbp), %eax
	sall	$6, %eax
	movl	%eax, %edx
	movl	-636(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-688(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	mulss	%xmm1, %xmm0
	movss	-616(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -616(%rbp)
	addl	$1, -624(%rbp)
.L56:
	movl	-624(%rbp), %eax
	cmpl	-628(%rbp), %eax
	jle	.L57
	movl	-628(%rbp), %eax
	sall	$6, %eax
	movl	%eax, %edx
	movl	-636(%rbp), %eax
	addl	%edx, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-696(%rbp), %rax
	addq	%rdx, %rax
	movss	-616(%rbp), %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -628(%rbp)
.L55:
	cmpl	$4095, -628(%rbp)
	jle	.L58
	addl	$1, -636(%rbp)
.L48:
	cmpl	$63, -636(%rbp)
	jle	.L59
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L60
	call	__stack_chk_fail@PLT
.L60:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	s4d_layer, .-s4d_layer
	.globl	gelu_inplace
	.type	gelu_inplace, @function
gelu_inplace:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	$0, -12(%rbp)
	jmp	.L62
.L63:
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -8(%rbp)
	movss	-8(%rbp), %xmm1
	movss	.LC6(%rip), %xmm0
	mulss	%xmm1, %xmm0
	mulss	-8(%rbp), %xmm0
	mulss	-8(%rbp), %xmm0
	movaps	%xmm0, %xmm1
	addss	-8(%rbp), %xmm1
	movss	.LC7(%rip), %xmm0
	mulss	%xmm1, %xmm0
	movss	%xmm0, -4(%rbp)
	movss	-8(%rbp), %xmm1
	movss	.LC8(%rip), %xmm0
	mulss	%xmm0, %xmm1
	movss	%xmm1, -32(%rbp)
	movl	-4(%rbp), %eax
	movd	%eax, %xmm0
	call	tanhf@PLT
	movss	.LC4(%rip), %xmm1
	addss	%xmm1, %xmm0
	movl	-12(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	mulss	-32(%rbp), %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -12(%rbp)
.L62:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jl	.L63
	nop
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	gelu_inplace, .-gelu_inplace
	.globl	softmax_inplace
	.type	softmax_inplace, @function
softmax_inplace:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	-40(%rbp), %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -24(%rbp)
	movl	$1, -28(%rbp)
	jmp	.L65
.L68:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	comiss	-24(%rbp), %xmm0
	jbe	.L66
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	%xmm0, -24(%rbp)
.L66:
	addl	$1, -28(%rbp)
.L65:
	movl	-28(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jl	.L68
	pxor	%xmm0, %xmm0
	movss	%xmm0, -20(%rbp)
	movl	$0, -28(%rbp)
	jmp	.L69
.L70:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	subss	-24(%rbp), %xmm0
	movd	%xmm0, %eax
	movl	-28(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-40(%rbp), %rdx
	leaq	(%rcx,%rdx), %rbx
	movd	%eax, %xmm0
	call	expf@PLT
	movd	%xmm0, %eax
	movl	%eax, (%rbx)
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movss	-20(%rbp), %xmm1
	addss	%xmm1, %xmm0
	movss	%xmm0, -20(%rbp)
	addl	$1, -28(%rbp)
.L69:
	movl	-28(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jl	.L70
	movl	$0, -28(%rbp)
	jmp	.L71
.L72:
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movl	-28(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	divss	-20(%rbp), %xmm0
	movss	%xmm0, (%rax)
	addl	$1, -28(%rbp)
.L71:
	movl	-28(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jl	.L72
	nop
	nop
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	softmax_inplace, .-softmax_inplace
	.globl	take_last_timestep
	.type	take_last_timestep, @function
take_last_timestep:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	leaq	1048320(%rax), %rcx
	movq	-16(%rbp), %rax
	movl	$256, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	memcpy@PLT
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	take_last_timestep, .-take_last_timestep
	.globl	argmax
	.type	argmax, @function
argmax:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movl	$0, -8(%rbp)
	movl	$1, -4(%rbp)
	jmp	.L76
.L79:
	movl	-4(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	movl	-8(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm1
	comiss	%xmm1, %xmm0
	jbe	.L77
	movl	-4(%rbp), %eax
	movl	%eax, -8(%rbp)
.L77:
	addl	$1, -4(%rbp)
.L76:
	movl	-4(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jl	.L79
	movl	-8(%rbp), %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	argmax, .-argmax
	.globl	forward
	.type	forward, @function
forward:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rax
	leaq	buf_hilbert(%rip), %rdx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	hilbert_scan
	movq	-8(%rbp), %rax
	leaq	16640(%rax), %rsi
	movq	-8(%rbp), %rax
	addq	$16384, %rax
	pushq	$4096
	movl	$64, %r9d
	movl	$1, %r8d
	leaq	buf_proj(%rip), %rdx
	movq	%rdx, %rcx
	leaq	buf_hilbert(%rip), %rdx
	movq	%rax, %rdi
	call	linear_layer
	addq	$8, %rsp
	movq	-8(%rbp), %rax
	leaq	49920(%rax), %r8
	movq	-8(%rbp), %rax
	leaq	33536(%rax), %rcx
	movq	-8(%rbp), %rax
	leaq	25344(%rax), %rdx
	movq	-8(%rbp), %rax
	leaq	17152(%rax), %rsi
	movq	-8(%rbp), %rax
	addq	$16896, %rax
	subq	$8, %rsp
	leaq	buf_s4d1(%rip), %rdi
	pushq	%rdi
	leaq	buf_proj(%rip), %r9
	movq	%rax, %rdi
	call	s4d_layer
	addq	$16, %rsp
	movl	$262144, %esi
	leaq	buf_s4d1(%rip), %rax
	movq	%rax, %rdi
	call	gelu_inplace
	movq	-8(%rbp), %rax
	leaq	83200(%rax), %r8
	movq	-8(%rbp), %rax
	leaq	66816(%rax), %rcx
	movq	-8(%rbp), %rax
	leaq	58624(%rax), %rdx
	movq	-8(%rbp), %rax
	leaq	50432(%rax), %rsi
	movq	-8(%rbp), %rax
	addq	$50176, %rax
	subq	$8, %rsp
	leaq	buf_s4d2(%rip), %rdi
	pushq	%rdi
	leaq	buf_s4d1(%rip), %r9
	movq	%rax, %rdi
	call	s4d_layer
	addq	$16, %rsp
	movl	$262144, %esi
	leaq	buf_s4d2(%rip), %rax
	movq	%rax, %rdi
	call	gelu_inplace
	leaq	buf_pooled(%rip), %rax
	movq	%rax, %rsi
	leaq	buf_s4d2(%rip), %rax
	movq	%rax, %rdi
	call	take_last_timestep
	movq	-8(%rbp), %rax
	leaq	84480(%rax), %rsi
	movq	-8(%rbp), %rax
	addq	$83456, %rax
	subq	$8, %rsp
	pushq	$1
	movl	$4, %r9d
	movl	$64, %r8d
	leaq	buf_logits(%rip), %rdx
	movq	%rdx, %rcx
	leaq	buf_pooled(%rip), %rdx
	movq	%rax, %rdi
	call	linear_layer
	addq	$16, %rsp
	movq	buf_logits(%rip), %rax
	movq	8+buf_logits(%rip), %rdx
	movq	-24(%rbp), %rcx
	movq	%rax, (%rcx)
	movq	%rdx, 8(%rcx)
	movq	-24(%rbp), %rax
	movl	$4, %esi
	movq	%rax, %rdi
	call	softmax_inplace
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	forward, .-forward
	.section	.rodata
	.align 16
.LC3:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.align 4
.LC4:
	.long	1065353216
	.align 4
.LC6:
	.long	1027024659
	.align 4
.LC7:
	.long	1061962282
	.align 4
.LC8:
	.long	1056964608
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
