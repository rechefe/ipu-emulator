# probe_signed_blt_int8: timing probe signed_blt in INT8
SET LR0 CR2 ;;
    BLT.S LR0 CR0 taken ;;
    SET LR7 CR3 ;;
    HALT ;;
taken:
    SET LR7 CR4 ;;
    HALT ;;
