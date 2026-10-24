# probe_signed_bge_nx8: timing probe signed_bge in NX8
SET LR0 CR2 ;;
    BGE.S LR0 CR0 taken ;;
    SET LR7 CR3 ;;
    HALT ;;
taken:
    SET LR7 CR4 ;;
    HALT ;;
