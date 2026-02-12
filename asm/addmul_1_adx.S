# addmul_1_adx.S - ADX/BMI2 addmul_1, 2-way unrolled (System V x86-64 ABI)
#
# uint64_t zint_mpn_addmul_1_adx(uint64_t* rp, const uint64_t* ap,
#                                  uint32_t n, uint64_t b);
#
# System V: rdi=rp, rsi=ap, edx=n, rcx=b
# Callee-saved: rbx, r12-r15, rbp (only rbx used here)
#
# 2-way with `loop` (preserves CF/OF for ADX dual-carry chain).

.intel_syntax noprefix

.text
.globl zint_mpn_addmul_1_adx
.type  zint_mpn_addmul_1_adx, @function

zint_mpn_addmul_1_adx:
    push rbx

    # Shuffle: need rdx=b for MULX, ecx=n for loop.
    # Input: edx=n, rcx=b. Swap via r8.
    mov  r8d, edx                # r8d = n
    mov  rdx, rcx                # rdx = b (MULX source)
    mov  ecx, r8d                # ecx = n

    test ecx, ecx
    jz   .Lzero

    # pair_count = (n - 1) / 2
    lea  r10d, [ecx-1]
    shr  r10d, 1

    # Clear CF/OF and set rax = 0 for ADOX.
    xor  eax, eax

    # If n is even, do one limb first to make remaining count odd.
    test ecx, 1
    jnz  .Lpriming

.Lpre_even:
    mulx r9, r8, qword ptr [rsi]
    adox r8, rax
    adcx r8, qword ptr [rdi]
    mov  qword ptr [rdi], r8
    mov  rax, r9

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]

.Lpriming:
    mulx r9, r8, qword ptr [rsi]
    adox r8, rax
    adcx r8, qword ptr [rdi]
    mov  qword ptr [rdi], r8

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]
    mov  ecx, r10d
    jrcxz .Lafter_loop

.Lloop:
    mulx r11, r10, qword ptr [rsi]
    adox r10, r9
    adcx r10, qword ptr [rdi]
    mov  qword ptr [rdi], r10

    mulx r9, r8, qword ptr [rsi+8]
    adox r8, r11
    adcx r8, qword ptr [rdi+8]
    mov  qword ptr [rdi+8], r8

    lea  rsi, [rsi+16]
    lea  rdi, [rdi+16]
    loop .Lloop

.Lafter_loop:
    mov  ebx, 0
    adox r9, rbx
    adcx r9, rbx
    mov  rax, r9

.Lzero:
    pop  rbx
    ret

.size zint_mpn_addmul_1_adx, .-zint_mpn_addmul_1_adx
