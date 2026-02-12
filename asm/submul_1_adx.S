# submul_1_adx.S - BMI2 accelerated mpn_submul_1 (System V x86-64 ABI)
#
# uint64_t zint_mpn_submul_1_adx(uint64_t* rp, const uint64_t* ap,
#                                  uint32_t n, uint64_t b);
#
# System V: rdi=rp, rsi=ap, edx=n, rcx=b
# Callee-saved: rbx (only one used here)
#
# Cannot use ADX dual-flag trick because SUB clobbers OF.
# Uses MULX + ADD/ADC for product carry chain + SUB/ADC for subtraction.

.intel_syntax noprefix

.text
.globl zint_mpn_submul_1_adx
.type  zint_mpn_submul_1_adx, @function

zint_mpn_submul_1_adx:
    # Shuffle: need rdx=b for MULX, ecx=n.
    mov  r8d, edx
    mov  rdx, rcx
    mov  ecx, r8d

    xor  eax, eax
    test ecx, ecx
    jz   .Ldone

    # pair_count = (n - 1) / 2
    lea  r10d, [ecx-1]
    shr  r10d, 1

    # If n is even, handle one limb first.
    test ecx, 1
    jnz  .Lpriming

.Lpre_even:
    mulx r9, r8, qword ptr [rsi]
    sub  qword ptr [rdi], r8
    adc  r9, 0
    mov  rax, r9

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]

.Lpriming:
    mulx r9, r8, qword ptr [rsi]
    add  r8, rax
    adc  r9, 0
    sub  qword ptr [rdi], r8
    adc  r9, 0
    mov  rax, r9

    lea  rsi, [rsi+8]
    lea  rdi, [rdi+8]
    mov  ecx, r10d
    jrcxz .Ldone

.Lloop:
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
    jnz  .Lloop

.Ldone:
    ret

.size zint_mpn_submul_1_adx, .-zint_mpn_submul_1_adx
