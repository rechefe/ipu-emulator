# probe_cond_blt_nx8: timing probe cond_blt in NX8
SET LR0 CR2 ; BLT.S LR0 CR1 taken ;;
    SET LR7 CR3 ;;
    HALT ;;
taken:
    SET LR7 CR4 ;;
    HALT ;;
