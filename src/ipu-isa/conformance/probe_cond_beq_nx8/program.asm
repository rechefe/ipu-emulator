# probe_cond_beq_nx8: timing probe cond_beq in NX8
SET LR0 CR2 ; BEQ LR0 CR0 taken ;;
    SET LR7 CR3 ;;
    HALT ;;
taken:
    SET LR7 CR4 ;;
    HALT ;;
