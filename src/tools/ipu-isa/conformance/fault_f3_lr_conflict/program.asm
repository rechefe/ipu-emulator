# fault_f3_lr_conflict: F-3: two writes of LR0 in one word
SET LR0 CR2 ; SET LR0 CR5 ;;
HALT ;;
