	.file	"nn.c"
	.text
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"rb"
.LC1:
	.string	"[ERROR] Cannot open: %s\n"
.LC2:
	.string	"[ERROR] Unexpected EOF in %s\n"
	.text
	.p2align 4
	.globl	load_weights
	.type	load_weights, @function
load_weights:
.LFB29:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	movq	%rdi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	movq	%rsi, %rbp
	leaq	.LC0(%rip), %rsi
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	call	fopen@PLT
	testq	%rax, %rax
	je	.L8
	movq	%rax, %rcx
	movl	$4096, %edx
	movl	$4, %esi
	movq	%rbp, %rdi
	movq	%rax, %rbx
	call	fread@PLT
	cmpq	$4096, %rax
	jne	.L4
	leaq	16384(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$64, %rax
	jne	.L4
	leaq	16640(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$64, %rax
	jne	.L4
	leaq	16896(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$64, %rax
	jne	.L4
	leaq	17152(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$2048, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$2048, %rax
	jne	.L4
	leaq	25344(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$2048, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$2048, %rax
	jne	.L4
	leaq	33536(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$4096, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$4096, %rax
	jne	.L4
	leaq	49920(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$64, %rax
	jne	.L4
	leaq	50176(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$64, %rax
	jne	.L4
	leaq	50432(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$2048, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$2048, %rax
	jne	.L4
	leaq	58624(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$2048, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$2048, %rax
	jne	.L4
	leaq	66816(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$4096, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$4096, %rax
	jne	.L4
	leaq	83200(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$64, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$64, %rax
	jne	.L4
	leaq	83456(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$256, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$256, %rax
	jne	.L4
	leaq	84480(%rbp), %rdi
	movq	%rbx, %rcx
	movl	$4, %edx
	movl	$4, %esi
	call	fread@PLT
	cmpq	$4, %rax
	jne	.L4
	movq	%rbx, %rdi
	call	fclose@PLT
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	xorl	%eax, %eax
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L4:
	.cfi_restore_state
	movq	stderr(%rip), %rdi
	movq	%r12, %rcx
	movl	$2, %esi
	xorl	%eax, %eax
	leaq	.LC2(%rip), %rdx
	call	__fprintf_chk@PLT
	movq	%rbx, %rdi
	call	fclose@PLT
.L3:
	popq	%rbx
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	movl	$-1, %eax
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
.L8:
	.cfi_restore_state
	movq	stderr(%rip), %rdi
	movq	%r12, %rcx
	movl	$2, %esi
	xorl	%eax, %eax
	leaq	.LC1(%rip), %rdx
	call	__fprintf_chk@PLT
	jmp	.L3
	.cfi_endproc
.LFE29:
	.size	load_weights, .-load_weights
	.p2align 4
	.globl	hilbert_scan
	.type	hilbert_scan, @function
hilbert_scan:
.LFB30:
	.cfi_startproc
	endbr64
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L10:
	movslq	(%rdi,%rax), %rcx
	movss	(%rsi,%rcx,4), %xmm0
	movss	%xmm0, (%rdx,%rax)
	addq	$4, %rax
	cmpq	$16384, %rax
	jne	.L10
	ret
	.cfi_endproc
.LFE30:
	.size	hilbert_scan, .-hilbert_scan
	.p2align 4
	.globl	linear_layer
	.type	linear_layer, @function
linear_layer:
.LFB31:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movl	56(%rsp), %eax
	testl	%eax, %eax
	jle	.L12
	movl	%r8d, %r10d
	movl	%r9d, %r14d
	movslq	%r9d, %r9
	movq	%rcx, %r8
	movslq	%r10d, %r15
	movq	%rdi, %rbx
	movq	%rsi, %r11
	movq	%rdx, %r12
	salq	$2, %r9
	leaq	0(,%r15,4), %rcx
	xorl	%r13d, %r13d
	xorl	%ebp, %ebp
	.p2align 4,,10
	.p2align 3
.L18:
	testl	%r14d, %r14d
	jle	.L14
	movq	%rbx, %rdx
	leaq	(%r12,%r13,4), %rsi
	xorl	%edi, %edi
	.p2align 4,,10
	.p2align 3
.L17:
	movss	(%r11,%rdi), %xmm1
	testl	%r10d, %r10d
	jle	.L15
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L16:
	movss	(%rdx,%rax), %xmm0
	mulss	(%rsi,%rax), %xmm0
	addq	$4, %rax
	addss	%xmm0, %xmm1
	cmpq	%rax, %rcx
	jne	.L16
.L15:
	movss	%xmm1, (%r8,%rdi)
	addq	$4, %rdi
	addq	%rcx, %rdx
	cmpq	%r9, %rdi
	jne	.L17
.L14:
	addl	$1, %ebp
	addq	%r9, %r8
	addq	%r15, %r13
	cmpl	%ebp, 56(%rsp)
	jne	.L18
.L12:
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE31:
	.size	linear_layer, .-linear_layer
	.p2align 4
	.globl	s4d_layer
	.type	s4d_layer, @function
s4d_layer:
.LFB32:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	movq	%rsi, %r14
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$424, %rsp
	.cfi_def_cfa_offset 480
	movq	480(%rsp), %rax
	movq	%rcx, 88(%rsp)
	xorl	%ecx, %ecx
	leaq	140(%rsp), %r13
	movq	%rdi, 64(%rsp)
	leaq	136(%rsp), %r12
	leaq	144(%rsp), %rbp
	movq	%rsi, 72(%rsp)
	leaq	272(%rsp), %rbx
	movq	%rdx, 80(%rsp)
	movq	%r8, 96(%rsp)
	movq	%r9, 104(%rsp)
	movq	%rax, 112(%rsp)
	movq	%fs:40, %rax
	movq	%rax, 408(%rsp)
	xorl	%eax, %eax
	movq	%rdx, 56(%rsp)
.L30:
	movq	64(%rsp), %rax
	movq	%rcx, 8(%rsp)
	movss	(%rax,%rcx,4), %xmm0
	call	expf@PLT
	movq	8(%rsp), %rcx
	movq	72(%rsp), %rdx
	movq	%r14, 128(%rsp)
	movq	%r12, 40(%rsp)
	movq	%rcx, %rax
	movq	%rcx, 120(%rsp)
	salq	$7, %rax
	movss	%xmm0, 24(%rsp)
	leaq	(%rdx,%rax), %r8
	movq	80(%rsp), %rdx
	movq	%r8, %r15
	addq	%rdx, %rax
	movq	%rcx, %rdx
	movq	%rax, 48(%rsp)
	movq	88(%rsp), %rax
	salq	$8, %rdx
	addq	%rax, %rdx
	xorl	%eax, %eax
	movq	%rax, %r14
	movq	%rdx, %r12
.L24:
	movss	(%r15,%r14), %xmm0
	call	expf@PLT
	movq	48(%rsp), %rax
	movq	40(%rsp), %rsi
	movq	%r13, %rdi
	movaps	%xmm0, %xmm7
	movss	%xmm0, 36(%rsp)
	movss	24(%rsp), %xmm0
	xorps	.LC4(%rip), %xmm7
	movss	(%rax,%r14), %xmm3
	movss	%xmm7, 8(%rsp)
	mulss	%xmm3, %xmm0
	movss	%xmm3, 32(%rsp)
	call	sincosf@PLT
	movss	24(%rsp), %xmm0
	mulss	8(%rsp), %xmm0
	movss	136(%rsp), %xmm3
	movss	140(%rsp), %xmm5
	movss	%xmm3, 20(%rsp)
	movss	%xmm5, 28(%rsp)
	call	expf@PLT
	movss	28(%rsp), %xmm5
	movss	20(%rsp), %xmm2
	movaps	%xmm0, %xmm4
	movss	(%r12,%r14,2), %xmm6
	movss	36(%rsp), %xmm1
	mulss	%xmm0, %xmm5
	movss	8(%rsp), %xmm7
	mulss	%xmm4, %xmm2
	subss	.LC5(%rip), %xmm2
	mulss	%xmm1, %xmm1
	movaps	%xmm5, %xmm0
	movss	4(%r12,%r14,2), %xmm5
	movaps	%xmm2, %xmm4
	mulss	%xmm6, %xmm4
	movaps	%xmm0, %xmm3
	mulss	%xmm5, %xmm3
	mulss	%xmm5, %xmm2
	mulss	%xmm6, %xmm0
	subss	%xmm3, %xmm4
	movss	32(%rsp), %xmm3
	addss	%xmm2, %xmm0
	movaps	%xmm3, %xmm2
	movaps	%xmm3, %xmm5
	mulss	%xmm3, %xmm2
	mulss	%xmm4, %xmm3
	mulss	%xmm0, %xmm5
	mulss	%xmm7, %xmm0
	addss	%xmm2, %xmm1
	movaps	%xmm7, %xmm2
	mulss	%xmm4, %xmm2
	subss	%xmm3, %xmm0
	addss	%xmm5, %xmm2
	divss	%xmm1, %xmm0
	divss	%xmm1, %xmm2
	movss	%xmm0, (%rbx,%r14)
	movss	%xmm2, 0(%rbp,%r14)
	addq	$4, %r14
	cmpq	$128, %r14
	jne	.L24
	movq	128(%rsp), %r14
	movq	40(%rsp), %r12
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L25:
	pxor	%xmm3, %xmm3
	xorl	%r15d, %r15d
	movq	%rax, 48(%rsp)
	pxor	%xmm2, %xmm2
	cvtsi2ssl	%eax, %xmm3
	movq	%r12, 40(%rsp)
	movq	%r14, %r12
	movq	%r15, %r14
	movq	56(%rsp), %r15
	movss	%xmm2, 8(%rsp)
	movss	%xmm3, 20(%rsp)
	.p2align 4,,10
	.p2align 3
.L26:
	movss	(%r12,%r14), %xmm0
	call	expf@PLT
	movq	40(%rsp), %rsi
	movq	%r13, %rdi
	movss	%xmm0, 36(%rsp)
	movss	24(%rsp), %xmm0
	mulss	(%r15,%r14), %xmm0
	mulss	20(%rsp), %xmm0
	call	sincosf@PLT
	movss	36(%rsp), %xmm1
	movss	24(%rsp), %xmm0
	movss	136(%rsp), %xmm2
	xorps	.LC4(%rip), %xmm1
	movss	140(%rsp), %xmm4
	mulss	%xmm1, %xmm0
	movss	%xmm2, 32(%rsp)
	mulss	20(%rsp), %xmm0
	movss	%xmm4, 28(%rsp)
	call	expf@PLT
	movss	32(%rsp), %xmm2
	movaps	%xmm0, %xmm1
	mulss	%xmm0, %xmm2
	movss	0(%rbp,%r14), %xmm0
	mulss	28(%rsp), %xmm1
	mulss	(%rbx,%r14), %xmm1
	addq	$4, %r14
	mulss	%xmm2, %xmm0
	subss	%xmm1, %xmm0
	addss	%xmm0, %xmm0
	addss	8(%rsp), %xmm0
	movss	%xmm0, 8(%rsp)
	cmpq	$128, %r14
	jne	.L26
	movq	48(%rsp), %rax
	leaq	s4d_kernel(%rip), %rcx
	movq	%r12, %r14
	movq	40(%rsp), %r12
	movss	%xmm0, (%rcx,%rax,4)
	addq	$1, %rax
	cmpq	$4096, %rax
	jne	.L25
	movq	120(%rsp), %rcx
	movq	96(%rsp), %rax
	leaq	4+s4d_kernel(%rip), %rsi
	xorl	%r9d, %r9d
	movss	(%rax,%rcx,4), %xmm2
	movq	104(%rsp), %rax
	leaq	0(,%rcx,4), %r8
	leaq	(%rax,%r8), %rdi
	movq	112(%rsp), %rax
	addq	%rax, %r8
	.p2align 4,,10
	.p2align 3
.L29:
	movss	(%rdi), %xmm1
	movq	%rdi, %rdx
	leaq	s4d_kernel(%rip), %rax
	mulss	%xmm2, %xmm1
	.p2align 4,,10
	.p2align 3
.L28:
	movss	(%rax), %xmm0
	mulss	(%rdx), %xmm0
	addq	$4, %rax
	subq	$256, %rdx
	addss	%xmm0, %xmm1
	cmpq	%rax, %rsi
	jne	.L28
	addl	$1, %r9d
	movss	%xmm1, (%r8)
	addq	$4, %rsi
	addq	$256, %rdi
	addq	$256, %r8
	cmpl	$4096, %r9d
	jne	.L29
	addq	$1, %rcx
	subq	$-128, 56(%rsp)
	subq	$-128, %r14
	cmpq	$64, %rcx
	jne	.L30
	movq	408(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L38
	addq	$424, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L38:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE32:
	.size	s4d_layer, .-s4d_layer
	.p2align 4
	.globl	gelu_inplace
	.type	gelu_inplace, @function
gelu_inplace:
.LFB33:
	.cfi_startproc
	endbr64
	testl	%esi, %esi
	jle	.L44
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movslq	%esi, %rsi
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	leaq	(%rdi,%rsi,4), %rbx
	subq	$24, %rsp
	.cfi_def_cfa_offset 48
	.p2align 4,,10
	.p2align 3
.L41:
	movss	0(%rbp), %xmm1
	movss	.LC6(%rip), %xmm0
	addq	$4, %rbp
	mulss	%xmm1, %xmm0
	movss	%xmm1, 12(%rsp)
	mulss	%xmm1, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm1, %xmm0
	mulss	.LC7(%rip), %xmm0
	call	tanhf@PLT
	movss	12(%rsp), %xmm1
	mulss	.LC8(%rip), %xmm1
	addss	.LC5(%rip), %xmm0
	mulss	%xmm0, %xmm1
	movss	%xmm1, -4(%rbp)
	cmpq	%rbp, %rbx
	jne	.L41
	addq	$24, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L44:
	.cfi_restore 3
	.cfi_restore 6
	ret
	.cfi_endproc
.LFE33:
	.size	gelu_inplace, .-gelu_inplace
	.p2align 4
	.globl	softmax_inplace
	.type	softmax_inplace, @function
softmax_inplace:
.LFB34:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	movq	%rdi, %rbx
	subq	$16, %rsp
	.cfi_def_cfa_offset 48
	movss	(%rdi), %xmm1
	cmpl	$1, %esi
	jle	.L48
	leal	-2(%rsi), %edx
	leaq	4(%rdi), %rax
	leaq	8(%rdi,%rdx,4), %rdx
	.p2align 4,,10
	.p2align 3
.L50:
	movss	(%rax), %xmm0
	addq	$4, %rax
	maxss	%xmm1, %xmm0
	movaps	%xmm0, %xmm1
	cmpq	%rdx, %rax
	jne	.L50
.L51:
	movslq	%esi, %rsi
	movq	%rbx, %rbp
	pxor	%xmm2, %xmm2
	leaq	(%rbx,%rsi,4), %r12
	.p2align 4,,10
	.p2align 3
.L53:
	movss	0(%rbp), %xmm0
	movss	%xmm2, 12(%rsp)
	addq	$4, %rbp
	movss	%xmm1, 8(%rsp)
	subss	%xmm1, %xmm0
	call	expf@PLT
	movss	12(%rsp), %xmm2
	movss	8(%rsp), %xmm1
	movss	%xmm0, -4(%rbp)
	cmpq	%r12, %rbp
	addss	%xmm0, %xmm2
	jne	.L53
	.p2align 4,,10
	.p2align 3
.L54:
	movss	(%rbx), %xmm0
	addq	$4, %rbx
	divss	%xmm2, %xmm0
	movss	%xmm0, -4(%rbx)
	cmpq	%r12, %rbx
	jne	.L54
	addq	$16, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L48:
	.cfi_restore_state
	je	.L51
	addq	$16, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE34:
	.size	softmax_inplace, .-softmax_inplace
	.p2align 4
	.globl	take_last_timestep
	.type	take_last_timestep, @function
take_last_timestep:
.LFB35:
	.cfi_startproc
	endbr64
	movdqu	1048320(%rdi), %xmm0
	movups	%xmm0, (%rsi)
	movdqu	1048336(%rdi), %xmm1
	movups	%xmm1, 16(%rsi)
	movdqu	1048352(%rdi), %xmm2
	movups	%xmm2, 32(%rsi)
	movdqu	1048368(%rdi), %xmm3
	movups	%xmm3, 48(%rsi)
	movdqu	1048384(%rdi), %xmm4
	movups	%xmm4, 64(%rsi)
	movdqu	1048400(%rdi), %xmm5
	movups	%xmm5, 80(%rsi)
	movdqu	1048416(%rdi), %xmm6
	movups	%xmm6, 96(%rsi)
	movdqu	1048432(%rdi), %xmm7
	movups	%xmm7, 112(%rsi)
	movdqu	1048448(%rdi), %xmm0
	movups	%xmm0, 128(%rsi)
	movdqu	1048464(%rdi), %xmm1
	movups	%xmm1, 144(%rsi)
	movdqu	1048480(%rdi), %xmm2
	movups	%xmm2, 160(%rsi)
	movdqu	1048496(%rdi), %xmm3
	movups	%xmm3, 176(%rsi)
	movdqu	1048512(%rdi), %xmm4
	movups	%xmm4, 192(%rsi)
	movdqu	1048528(%rdi), %xmm5
	movups	%xmm5, 208(%rsi)
	movdqu	1048544(%rdi), %xmm6
	movups	%xmm6, 224(%rsi)
	movdqu	1048560(%rdi), %xmm7
	movups	%xmm7, 240(%rsi)
	ret
	.cfi_endproc
.LFE35:
	.size	take_last_timestep, .-take_last_timestep
	.p2align 4
	.globl	argmax
	.type	argmax, @function
argmax:
.LFB36:
	.cfi_startproc
	endbr64
	cmpl	$1, %esi
	jle	.L67
	movl	%esi, %esi
	movl	$1, %eax
	xorl	%edx, %edx
	.p2align 4,,10
	.p2align 3
.L66:
	movslq	%edx, %rcx
	movss	(%rdi,%rax,4), %xmm0
	ucomiss	(%rdi,%rcx,4), %xmm0
	cmova	%eax, %edx
	addq	$1, %rax
	cmpq	%rax, %rsi
	jne	.L66
	movl	%edx, %eax
	ret
	.p2align 4,,10
	.p2align 3
.L67:
	xorl	%edx, %edx
	movl	%edx, %eax
	ret
	.cfi_endproc
.LFE36:
	.size	argmax, .-argmax
	.p2align 4
	.globl	forward
	.type	forward, @function
forward:
.LFB37:
	.cfi_startproc
	endbr64
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	movq	%rsi, %rcx
	xorl	%eax, %eax
	leaq	buf_hilbert(%rip), %rsi
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movq	%rdx, %rbx
	subq	$24, %rsp
	.cfi_def_cfa_offset 64
	.p2align 4,,10
	.p2align 3
.L71:
	movslq	0(%rbp,%rax), %rdx
	movss	(%rcx,%rdx,4), %xmm0
	movss	%xmm0, (%rsi,%rax)
	addq	$4, %rax
	cmpq	$16384, %rax
	jne	.L71
	leaq	buf_hilbert(%rip), %rcx
	leaq	buf_proj(%rip), %rdx
	leaq	16384(%rcx), %rsi
	.p2align 4,,10
	.p2align 3
.L73:
	movss	(%rcx), %xmm1
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L72:
	movss	16384(%rbp,%rax), %xmm0
	mulss	%xmm1, %xmm0
	addss	16640(%rbp,%rax), %xmm0
	movss	%xmm0, (%rdx,%rax)
	addq	$4, %rax
	cmpq	$256, %rax
	jne	.L72
	addq	$4, %rcx
	addq	$256, %rdx
	cmpq	%rcx, %rsi
	jne	.L73
	subq	$8, %rsp
	.cfi_def_cfa_offset 72
	leaq	buf_s4d1(%rip), %r13
	leaq	33536(%rbp), %rcx
	pushq	%r13
	.cfi_def_cfa_offset 80
	leaq	17152(%rbp), %rsi
	leaq	25344(%rbp), %rdx
	leaq	16896(%rbp), %rdi
	leaq	buf_proj(%rip), %r9
	leaq	49920(%rbp), %r8
	leaq	1048576(%r13), %r12
	call	s4d_layer
	popq	%rcx
	.cfi_def_cfa_offset 72
	popq	%rsi
	.cfi_def_cfa_offset 64
	.p2align 4,,10
	.p2align 3
.L74:
	movss	0(%r13), %xmm1
	movss	.LC6(%rip), %xmm0
	addq	$4, %r13
	mulss	%xmm1, %xmm0
	movss	%xmm1, 8(%rsp)
	mulss	%xmm1, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm1, %xmm0
	mulss	.LC7(%rip), %xmm0
	call	tanhf@PLT
	movss	8(%rsp), %xmm1
	mulss	.LC8(%rip), %xmm1
	addss	.LC5(%rip), %xmm0
	mulss	%xmm0, %xmm1
	movss	%xmm1, -4(%r13)
	cmpq	%r13, %r12
	jne	.L74
	subq	$8, %rsp
	.cfi_def_cfa_offset 72
	leaq	buf_s4d2(%rip), %r13
	leaq	58624(%rbp), %rdx
	pushq	%r13
	.cfi_def_cfa_offset 80
	leaq	66816(%rbp), %rcx
	leaq	50432(%rbp), %rsi
	leaq	50176(%rbp), %rdi
	leaq	buf_s4d1(%rip), %r9
	leaq	83200(%rbp), %r8
	leaq	1048576(%r13), %r12
	call	s4d_layer
	popq	%rax
	.cfi_def_cfa_offset 72
	popq	%rdx
	.cfi_def_cfa_offset 64
	.p2align 4,,10
	.p2align 3
.L75:
	movss	0(%r13), %xmm1
	movss	.LC6(%rip), %xmm0
	addq	$4, %r13
	mulss	%xmm1, %xmm0
	movss	%xmm1, 8(%rsp)
	mulss	%xmm1, %xmm0
	mulss	%xmm1, %xmm0
	addss	%xmm1, %xmm0
	mulss	.LC7(%rip), %xmm0
	call	tanhf@PLT
	movss	8(%rsp), %xmm1
	mulss	.LC8(%rip), %xmm1
	addss	.LC5(%rip), %xmm0
	mulss	%xmm0, %xmm1
	movss	%xmm1, -4(%r13)
	cmpq	%r13, %r12
	jne	.L75
	leaq	83456(%rbp), %rdx
	xorl	%esi, %esi
	leaq	buf_logits(%rip), %rdi
	movdqa	1048320+buf_s4d2(%rip), %xmm6
	movdqa	1048336+buf_s4d2(%rip), %xmm7
	movdqa	1048384+buf_s4d2(%rip), %xmm5
	leaq	buf_pooled(%rip), %rcx
	movaps	%xmm6, buf_pooled(%rip)
	movdqa	1048352+buf_s4d2(%rip), %xmm6
	movaps	%xmm7, 16+buf_pooled(%rip)
	movdqa	1048368+buf_s4d2(%rip), %xmm7
	movaps	%xmm6, 32+buf_pooled(%rip)
	movdqa	1048400+buf_s4d2(%rip), %xmm6
	movaps	%xmm7, 48+buf_pooled(%rip)
	movdqa	1048416+buf_s4d2(%rip), %xmm7
	movaps	%xmm5, 64+buf_pooled(%rip)
	movdqa	1048432+buf_s4d2(%rip), %xmm5
	movaps	%xmm6, 80+buf_pooled(%rip)
	movdqa	1048448+buf_s4d2(%rip), %xmm6
	movaps	%xmm7, 96+buf_pooled(%rip)
	movdqa	1048464+buf_s4d2(%rip), %xmm7
	movaps	%xmm5, 112+buf_pooled(%rip)
	movdqa	1048480+buf_s4d2(%rip), %xmm5
	movaps	%xmm6, 128+buf_pooled(%rip)
	movdqa	1048496+buf_s4d2(%rip), %xmm6
	movaps	%xmm7, 144+buf_pooled(%rip)
	movdqa	1048512+buf_s4d2(%rip), %xmm7
	movaps	%xmm5, 160+buf_pooled(%rip)
	movdqa	1048528+buf_s4d2(%rip), %xmm5
	movaps	%xmm6, 176+buf_pooled(%rip)
	movdqa	1048544+buf_s4d2(%rip), %xmm6
	movaps	%xmm7, 192+buf_pooled(%rip)
	movdqa	1048560+buf_s4d2(%rip), %xmm7
	movaps	%xmm5, 208+buf_pooled(%rip)
	movaps	%xmm6, 224+buf_pooled(%rip)
	movaps	%xmm7, 240+buf_pooled(%rip)
	.p2align 4,,10
	.p2align 3
.L77:
	movss	84480(%rbp,%rsi), %xmm1
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L76:
	movups	(%rdx,%rax), %xmm0
	mulps	(%rcx,%rax), %xmm0
	addq	$16, %rax
	addss	%xmm0, %xmm1
	movaps	%xmm0, %xmm2
	shufps	$85, %xmm0, %xmm2
	addss	%xmm1, %xmm2
	movaps	%xmm0, %xmm1
	unpckhps	%xmm0, %xmm1
	shufps	$255, %xmm0, %xmm0
	addss	%xmm2, %xmm1
	addss	%xmm0, %xmm1
	cmpq	$256, %rax
	jne	.L76
	movss	%xmm1, (%rdi,%rsi)
	addq	$4, %rsi
	addq	$256, %rdx
	cmpq	$16, %rsi
	jne	.L77
	movq	buf_logits(%rip), %rax
	movq	8+buf_logits(%rip), %rdx
	movq	%rbx, %rbp
	leaq	16(%rbx), %r12
	movl	$0x00000000, 8(%rsp)
	movd	%eax, %xmm1
	movq	%rax, (%rbx)
	shrq	$32, %rax
	movd	%eax, %xmm0
	movq	%rdx, 8(%rbx)
	maxss	%xmm1, %xmm0
	movd	%edx, %xmm1
	shrq	$32, %rdx
	movd	%edx, %xmm5
	maxss	%xmm0, %xmm1
	maxss	%xmm1, %xmm5
	movss	%xmm5, 12(%rsp)
	.p2align 4,,10
	.p2align 3
.L83:
	movss	0(%rbp), %xmm0
	subss	12(%rsp), %xmm0
	addq	$4, %rbp
	call	expf@PLT
	movss	%xmm0, -4(%rbp)
	addss	8(%rsp), %xmm0
	movss	%xmm0, 8(%rsp)
	cmpq	%r12, %rbp
	jne	.L83
	movups	(%rbx), %xmm1
	shufps	$0, %xmm0, %xmm0
	divps	%xmm0, %xmm1
	movups	%xmm1, (%rbx)
	addq	$24, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE37:
	.size	forward, .-forward
	.local	s4d_kernel
	.comm	s4d_kernel,16384,32
	.local	buf_logits
	.comm	buf_logits,16,16
	.local	buf_pooled
	.comm	buf_pooled,256,32
	.local	buf_s4d2
	.comm	buf_s4d2,1048576,32
	.local	buf_s4d1
	.comm	buf_s4d1,1048576,32
	.local	buf_proj
	.comm	buf_proj,1048576,32
	.local	buf_hilbert
	.comm	buf_hilbert,16384,32
	.section	.rodata.cst16,"aM",@progbits,16
	.align 16
.LC4:
	.long	-2147483648
	.long	0
	.long	0
	.long	0
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC5:
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
