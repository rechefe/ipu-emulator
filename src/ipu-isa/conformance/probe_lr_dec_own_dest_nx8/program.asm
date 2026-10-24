# probe_lr_dec_own_dest_nx8: timing probe lr_dec_own_dest in NX8
SET LR0 CR2 ;;
DEC LR0 3 ; ADD LR1 LR0 CR0 ;;
HALT ;;
