# fault_f6_xmem_row: F-6: an XMEM row past the end of XMEM
SET LR0 CR6 ;;
LD.MULT R0 LR0 CR7 ;;
HALT ;;
