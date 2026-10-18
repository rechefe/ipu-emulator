# fault_f9_disabled: F-9 disabled: the run passes the limit to HALT
SET LR1 CR8 ;;
loop:
INC LR0 1 ;;
BNE LR0 LR1 loop ;;
HALT ;;
