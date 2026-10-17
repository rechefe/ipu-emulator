# probe_break_ifeq_int8: timing probe break_ifeq in INT8
SET LR0 CR2 ; BREAK.IFEQ LR0 5 ;;
BREAK.IFEQ LR0 5 ;;
HALT ;;
