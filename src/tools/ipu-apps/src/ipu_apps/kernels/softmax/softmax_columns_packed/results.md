# softmax_columns_packed benchmark

```
config                        cycles  cyc/rows  mult%   acc%  lanes%  ident%  effMAC%
-------------------------------------------------------------------------------------
rows=64,width=8                  234      3.66  24.4%  24.4%   24.4%   56.1%    10.7%
rows=64,width=16                 234      3.66  24.4%  24.4%   24.4%   56.1%    10.7%
rows=100,width=32                525      5.25  25.5%  25.5%   25.5%   43.3%    14.5%
rows=128,width=64               1250      9.77  26.0%  26.0%   26.0%   40.6%    15.4%

ISA aliases: A1 MOV_RC  A2 ADD_VV  A3 SUB_VV  A4 AGG_RC  A7 REDUCE_SEG  A8 MOV_ACC  A9 EXP  A10 FRACTIONAL_SCALAR  A11 ACC_MUL  A12 ADDRESSING  A13 NARROW_MULT  A14 MULTIPLE_ACC  A15 SELECT
config                        A1  A2  A3  A4  A7  A8  A9  A10  A11  A12  A13  A14  A15
--------------------------------------------------------------------------------------
rows=64,width=8                9   7   8   8   2  13   8   16    1   85   57   33    1
rows=64,width=16               9   7   8   8   2  13   8   16    1   85   57   33    0
rows=100,width=32             26   3  25   4   2  30  25   50    1  213  134  101    0
rows=128,width=64             65   1  64   2   2  69  64  128    1  521  325  257    0
```

mult% / acc%: cycles the stage was busy. lanes%: multiplier lanes that carried data. ident%: busy multiply cycles that were identity multiplies (data routing, not MACs). effMAC%: peak MACs actually retired. ISA aliases: verified occurrences of each alias in docs/content/isa-alias-catalogue.md, all subtypes summed (+N? = unverified candidates); occurrences can overlap, so they are not savings estimates.
