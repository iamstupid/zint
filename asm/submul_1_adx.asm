; submul_1_adx.asm - BMI2 accelerated mpn_submul_1
;
; Implements:
;   rp[0..n) -= ap[0..n) * b
;   returns carry/borrow limb
;
; Signature (Windows x64):
;   uint64_t zint_mpn_submul_1_adx(uint64_t* rp, const uint64_t* ap, uint32_t n, uint64_t b);
;
; Requires: BMI2 (MULX).
; Unlike addmul_1, cannot use ADX dual-flag trick because SUB clobbers OF.
; Uses MULX + ADD/ADC for product carry chain + SUB/ADC for subtraction.

option casemap:none

.code

PUBLIC zint_mpn_submul_1_adx
zint_mpn_submul_1_adx PROC
    ; rcx = rp, rdx = ap, r8d = n, r9 = b
    push rbx
    push rsi
    push rdi

    mov  rdi, rcx                ; rp
    mov  rsi, rdx                ; ap
    mov  ecx, r8d                ; n
    mov  rdx, r9                 ; b (mulx uses rdx)

    xor  eax, eax                ; carry = 0
    test ecx, ecx
    jz   Ldone

    ; pair_count = (n - 1) / 2
    lea  r10d, [ecx-1]
    shr  r10d, 1

    ; If n is even, handle one limb first to make remaining count odd.
    test ecx, 1
    jnz  Lpriming

Lpre_even:
    mulx r9, r8, qword ptr [rsi]
    sub  qword ptr [rdi], r8
    adc  r9, 0
    mov  rax, r9

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]

Lpriming:
    mulx r9, r8, qword ptr [rsi]
    add  r8, rax
    adc  r9, 0
    sub  qword ptr [rdi], r8
    adc  r9, 0
    mov  rax, r9

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]
    mov  ecx, r10d
    jrcxz Ldone

Lloop:
    ; Process 2 limbs per iteration.
    mulx r11, r10, qword ptr [rsi]
    add  r10, rax
    adc  r11, 0
    sub  qword ptr [rdi], r10
    adc  r11, 0

    mulx r9, r8, qword ptr [rsi+8]
    add  r8, r11
    adc  r9, 0
    sub  qword ptr [rdi+8], r8
    adc  r9, 0
    mov  rax, r9

    lea  rsi, [rsi+16]
    lea  rdi, [rdi+16]
    dec  ecx
    jnz  Lloop

Ldone:
    pop  rdi
    pop  rsi
    pop  rbx
    ret
zint_mpn_submul_1_adx ENDP

END
