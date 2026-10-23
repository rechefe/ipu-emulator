# probe_cond_bne_nx8: timing probe cond_bne in NX8
SET LR0 CR2 ; BNE LR0 CR0 taken ;;
    SET LR7 CR3 ;;
    HALT ;;
taken:
    SET LR7 CR4 ;;
    HALT ;;
