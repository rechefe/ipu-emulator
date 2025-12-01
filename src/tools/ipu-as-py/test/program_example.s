# Start instruction
start:
    beq    lr13 lr15 end; // branch to end if lr13 == lr15
    mac.ee rq4 r1 r2;
    ldr    lr3 cr9;
    ;;

end:
    mac.ev rq8 r5 r9 lr0;
    beq    lr0 lr1 +2;
    incr   lr2 15;
    ;;

    b      start; mac.ev rq0 r3 mem_bypass lr15;
    ;;