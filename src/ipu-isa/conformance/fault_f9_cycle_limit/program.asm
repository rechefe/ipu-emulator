# fault_f9_cycle_limit: F-9: the cycle limit
SET LR1 CR8 ;;
loop:
INC LR0 1 ;;
BNE LR0 LR1 loop ;;
HALT ;;
