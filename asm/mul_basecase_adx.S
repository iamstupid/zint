# mul_basecase_adx.S - Fused schoolbook multiply, ADX/BMI2 (System V x86-64 ABI)
#
# void zint_mpn_mul_basecase_adx(uint64_t* rp, const uint64_t* ap, uint32_t an,
#                                  const uint64_t* bp, uint32_t bn);
#
# System V: rdi=rp, rsi=ap, edx=an, rcx=bp, r8d=bn
# Callee-saved: rbx, r12-r15, rbp
#
# Strategy:
#   1. Zero rp[0..an+bn) via rep stosq
#   2. For j = 0..bn-1: inner 2-way addmul_1(rp+j, ap, an, bp[j])

.intel_syntax noprefix

.text
.globl zint_mpn_mul_basecase_adx
.type  zint_mpn_mul_basecase_adx, @function

zint_mpn_mul_basecase_adx:
    push rbx
    push r12
    push r13
    push r14
    push r15
    push rbp

    # Save inputs into callee-saved regs for row loop.
    # rdi=rp (will be modified by rep stosq, save it)
    mov  r15, rdi                # r15 = rp_base
    mov  r12, rsi                # r12 = ap (preserved across rows)
    mov  r13d, edx               # r13d = an
    mov  r14, rcx                # r14 = bp
    mov  ebp, r8d                # ebp = bn (row counter)

    # ===== Step 1: Zero rp[0..an+bn) using rep stosq =====
    # rep stosq uses rdi (dest), ecx (count), rax (value)
    lea  ecx, [r13d + ebp]      # ecx = an + bn
    xor  eax, eax
    rep  stosq
    mov  rdi, r15                # restore rp

    # ===== Row loop: j = 0..bn-1 =====
.Lrow:
    mov  rbx, rdi                # save row start in rbx
    mov  rsi, r12                # ap
    mov  ecx, r13d               # n = an
    mov  rdx, qword ptr [r14]   # rdx = bp[j] (multiplier for MULX)

    # pair_count = (n - 1) / 2
    lea  r10d, [ecx-1]
    shr  r10d, 1

    # Clear CF/OF and set rax = 0 for ADOX.
    xor  eax, eax

    # If n is even, do one limb first.
    test ecx, 1
    jnz  .Rpriming

.Rpre_even:
    mulx r9, r8, qword ptr [rsi]
    adox r8, rax
    adcx r8, qword ptr [rdi]
    mov  qword ptr [rdi], r8
    mov  rax, r9

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]

.Rpriming:
    mulx r9, r8, qword ptr [rsi]
    adox r8, rax
    adcx r8, qword ptr [rdi]
    mov  qword ptr [rdi], r8

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]
    mov  ecx, r10d
    jrcxz .Rafter_loop

.Rloop:
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
    loop .Rloop

.Rafter_loop:
    # Collect final carries into r9.
    mov  r8d, 0                  # do not clobber flags
    adox r9, r8
    adcx r9, r8

    # Store carry at rp[j + an].
    # rdi has advanced by an*8 from row start (rbx), so [rdi] == rp[j + an].
    mov  qword ptr [rdi], r9

    # Advance to next row
    lea  rdi, [rbx+8]           # rp pointer += 1 limb from row start
    lea  r14, [r14+8]           # bp pointer += 1
    dec  ebp                     # row counter--
    jnz  .Lrow

    pop  rbp
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    ret

.size zint_mpn_mul_basecase_adx, .-zint_mpn_mul_basecase_adx
