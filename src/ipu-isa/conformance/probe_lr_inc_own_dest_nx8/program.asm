# probe_lr_inc_own_dest_nx8: timing probe lr_inc_own_dest in NX8
SET LR0 CR2 ;;
INC LR0 3 ; ADD LR1 LR0 CR0 ;;
HALT ;;
