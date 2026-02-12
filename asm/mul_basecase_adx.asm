; mul_basecase_adx.asm - Fused schoolbook multiply using ADX/BMI2, 2-way inner loop
;
; Implements:
;   rp[0..an+bn) = ap[0..an) * bp[0..bn)
;
; Signature (Windows x64):
;   void zint_mpn_mul_basecase_adx(uint64_t* rp, const uint64_t* ap, uint32_t an,
;                                   const uint64_t* bp, uint32_t bn);
;
; Strategy:
;   1. Zero rp[0..an+bn)
;   2. For j = 0..bn-1: inner 2-way addmul_1(rp+j, ap, an, bp[j])
;      with carry stored at rp[j+an]
;
; Inner loop uses `loop` instruction to preserve CF/OF for ADX dual-carry chain.

option casemap:none

.code

PUBLIC zint_mpn_mul_basecase_adx
zint_mpn_mul_basecase_adx PROC
    ; Windows x64: rcx=rp, rdx=ap, r8d=an, r9=bp
    ; 5th param bn at [rsp+40] (before pushes)
    push rbx
    push rsi
    push rdi
    push r12
    push r13
    push r14
    push r15
    push rbp                     ; 8 pushes = 64 bytes

    mov  rdi, rcx                ; rp_base
    mov  r12, rdx                ; ap (saved across rows)
    mov  r13d, r8d               ; an (saved across rows)
    mov  r14, r9                 ; bp
    mov  r15d, dword ptr [rsp + 104]  ; bn (5th param: rsp + 40 + 64)

    ; ===== Step 1: Zero rp[0..an+bn) using rep stosq =====
    mov  r8, rdi                 ; save rp_base in r8
    lea  ecx, [r13d + r15d]     ; ecx = an + bn (count for stosq, zero-extended)
    xor  eax, eax               ; value = 0
    rep  stosq                   ; zero rp[0..an+bn)
    mov  rdi, r8                 ; restore rp_base

    ; ===== Row loop: j = 0..bn-1 =====
Lrow:
    mov  rbp, rdi                ; save row start
    mov  rsi, r12                ; ap
    mov  ecx, r13d               ; n = an
    mov  rdx, qword ptr [r14]   ; rdx = bp[j] (multiplier for MULX)

    ; pair_count = (n - 1) / 2  (must compute BEFORE clearing flags; shr clobbers OF)
    lea  r10d, [ecx-1]
    shr  r10d, 1

    ; Clear CF/OF and set rax = 0 for ADOX.
    xor  eax, eax

    ; If n is even, do one limb first to make remaining count odd.
    test ecx, 1
    jnz  Rpriming

Rpre_even:
    mulx r9, r8, qword ptr [rsi]
    adox r8, rax
    adcx r8, qword ptr [rdi]
    mov  qword ptr [rdi], r8
    mov  rax, r9                 ; carry hi forward

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]

Rpriming:
    ; Prime previous-hi in r9.
    mulx r9, r8, qword ptr [rsi]
    adox r8, rax
    adcx r8, qword ptr [rdi]
    mov  qword ptr [rdi], r8

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]
    mov  ecx, r10d
    jrcxz Rafter_loop

Rloop:
    ; 2 limbs per iteration, preserving CF/OF.
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
    loop Rloop                   ; loop preserves CF/OF

Rafter_loop:
    ; Collect final carries into r9.
    mov  ebx, 0                  ; do not clobber flags
    adox r9, rbx
    adcx r9, rbx

    ; Store carry at rp[j + an].
    ; rdi has advanced by an*8 from row start (rbp), so [rdi] == rp[j + an].
    ; But we need to verify: rdi started at rbp, advanced 8 per priming limb(s),
    ; then 16 per loop pair. Total advance = an * 8. But rdi was also advanced
    ; by 8 in priming. Let me trace:
    ;   After pre_even (if even): rdi += 8 (1 limb)
    ;   After priming: rdi += 8 (1 limb)
    ;   After loop: rdi += 16 * pair_count
    ;
    ; If n is odd:  1 + 2*((n-1)/2) = 1 + n - 1 = n  limbs advanced. OK.
    ; If n is even: 1 + 1 + 2*((n-1)/2) = 2 + 2*((n-2)/2) = 2 + n - 2 = n. OK.
    ;
    ; So rdi = rbp + an*8, meaning [rdi] is rp[j + an]. Store carry there.
    mov  qword ptr [rdi], r9

    ; Advance to next row
    lea  rdi, [rbp+8]           ; rp pointer += 1 limb from row start
    lea  r14, [r14+8]           ; bp pointer += 1
    dec  r15d                    ; row counter--
    jnz  Lrow

    ; ===== Done =====
    pop  rbp
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rdi
    pop  rsi
    pop  rbx
    ret
zint_mpn_mul_basecase_adx ENDP

END
