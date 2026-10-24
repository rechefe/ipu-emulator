# probe_cond_br_nx8: timing probe cond_br in NX8
SET LR0 CR5 ;;
        SET LR0 CR6 ; BR LR0 ;;
        HALT ;;
        HALT ;;
        SET LR7 CR3 ;;
        HALT ;;
        SET LR7 CR4 ;;
        HALT ;;
