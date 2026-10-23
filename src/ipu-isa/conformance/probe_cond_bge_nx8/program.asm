# probe_cond_bge_nx8: timing probe cond_bge in NX8
SET LR0 CR2 ; BGE.S LR0 CR1 taken ;;
    SET LR7 CR3 ;;
    HALT ;;
taken:
    SET LR7 CR4 ;;
    HALT ;;
