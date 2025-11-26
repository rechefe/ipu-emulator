import pytest
from ipu_as import program, label


def test_program_encode():
    program_str = """
    # Hello world
    start: incr lr0 10; ldr lr1 cr3; macu rq4 r1 r2 lr4; beq lr3 lr4 end  # Stupid fucking comment
    set lr8 100; str lr9 cr5; mac r1 r2 r3 lr6; bkpt lr0 lr0 start
    
    end: add lr15 200; ldr lr9 cr15; mac rq0 r4 r5 lr7; b lr1 lr2 end
    """
    prog = program.Program(program_str)
    encoded = prog.encode()
    assert len(encoded) == 3
    assert label.ipu_labels.get_address("start") == 0
    assert label.ipu_labels.get_address("end") == 2
