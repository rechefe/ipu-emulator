# probe_cond_beq_int8: timing probe cond_beq in INT8
SET LR0 CR2 ; BEQ LR0 CR0 taken ;;
    SET LR7 CR3 ;;
    HALT ;;
taken:
    SET LR7 CR4 ;;
    HALT ;;
