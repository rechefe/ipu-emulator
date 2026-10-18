# probe_lr_subslot_exception_int8: timing probe lr_subslot_exception in INT8
# program.bin edits the assembled words: Point LR sub-slot 1's SET source at a CR index that does not exist.
SET LR1 CR2 ; SET LR2 CR3 ;;
HALT ;;
