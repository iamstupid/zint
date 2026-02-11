; addmul_1_adx.asm - ADX/BMI2 accelerated mpn_addmul_1
;
; Implements:
;   rp[0..n) += ap[0..n) * b
;   returns carry limb
;
; Signature (Windows x64):
;   uint64_t zint_mpn_addmul_1_adx(uint64_t* rp, const uint64_t* ap, uint32_t n, uint64_t b);
;
; Requires: BMI2 (MULX) + ADX (ADCX/ADOX).
; Caller must CPUID-gate before calling on CPUs without ADX.

option casemap:none

.code

PUBLIC zint_mpn_addmul_1_adx
zint_mpn_addmul_1_adx PROC
    ; rcx = rp, rdx = ap, r8d = n, r9 = b
    push rbx
    push rsi
    push rdi

    mov  rdi, rcx                ; rp
    mov  rsi, rdx                ; ap
    mov  ecx, r8d                ; n
    mov  rdx, r9                 ; b (mulx uses rdx)

    test ecx, ecx
    jz   Lzero

    ; pair_count = (n - 1) / 2  (number of 2-limb iterations after priming)
    lea  r10d, [ecx-1]
    shr  r10d, 1

    ; Clear CF/OF and set rax = 0 for ADOX.
    xor  eax, eax

    ; If n is even, do one iteration to make (n-1) odd so that after priming
    ; the remaining count is even (processed in 2-limb loop iterations).
    test ecx, 1
    jnz  Lpriming

Lpre_even:
    mulx r9, r8, qword ptr [rsi]
    adox r8, rax
    adcx r8, qword ptr [rdi]
    mov  qword ptr [rdi], r8

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]

Lpriming:
    ; Prime previous-hi in r9.
    mulx r9, r8, qword ptr [rsi]
    adox r8, rax
    adcx r8, qword ptr [rdi]
    mov  qword ptr [rdi], r8

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]
    mov  ecx, r10d
    jrcxz Lafter_loop

Lloop:
    ; Process 2 limbs per iteration, preserving CF/OF across the loop.
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
    loop Lloop

Lafter_loop:
    ; carry_out = last_hi + final_OF + final_CF
    mov  ebx, 0                  ; do not clobber flags
    adox r9, rbx
    adcx r9, rbx
    mov  rax, r9

Lzero:
    pop  rdi
    pop  rsi
    pop  rbx
    ret
zint_mpn_addmul_1_adx ENDP

END
