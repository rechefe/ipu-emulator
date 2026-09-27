# softmax_columns benchmark

```
config                        cycles  cyc/rows  mult%   acc%  lanes%  ident%  effMAC%
-------------------------------------------------------------------------------------
rows=16,width=128                327     20.44  24.5%  24.5%   24.5%   40.0%    14.7%
rows=64,width=128               1239     19.36  25.8%  25.8%   25.8%   40.0%    15.5%
rows=128,width=128              2455     19.18  26.1%  26.1%   26.1%   40.0%    15.6%
rows=32,width=256               1252     39.12  25.6%  25.6%   25.6%   40.0%    15.3%
rows=64,width=384               3697     57.77  26.0%  26.0%   26.0%   40.0%    15.6%

ISA aliases: A1 MOV_RC  A3 SUB_VV  A8 MOV_ACC  A9 EXP  A10 FRACTIONAL_SCALAR  A12 ADDRESSING  A13 NARROW_MULT  A14 MULTIPLE_ACC
config                         A1   A3   A8   A9  A10   A12  A13  A14
---------------------------------------------------------------------
rows=16,width=128              16   16   16   16   32   128   80   63
rows=64,width=128              64   64   64   64  128   512  320  255
rows=128,width=128            128  128  128  128  256  1024  640  511
rows=32,width=256              64   64   64   64  128   513  320  254
rows=64,width=384             192  192  192  192  384  1538  960  765
```

mult% / acc%: cycles the stage was busy. lanes%: multiplier lanes that carried data. ident%: busy multiply cycles that were identity multiplies (data routing, not MACs). effMAC%: peak MACs actually retired. ISA aliases: verified occurrences of each alias in docs/content/isa-alias-catalogue.md, all subtypes summed (+N? = unverified candidates); occurrences can overlap, so they are not savings estimates.
