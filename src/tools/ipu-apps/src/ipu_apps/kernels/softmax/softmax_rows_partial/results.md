# softmax_rows_partial benchmark

```
config                        cycles  cyc/rows  mult%   acc%  lanes%  ident%  effMAC%
-------------------------------------------------------------------------------------
n=8,rows=16                      316     19.75  25.3%  25.3%    1.6%   40.0%     0.9%
n=16,rows=16                     316     19.75  25.3%  25.3%    3.2%   40.0%     1.9%
n=32,rows=40                     834     20.85  24.0%  24.0%    6.0%   40.0%     3.6%
n=64,rows=50                    1174     23.48  21.3%  21.3%   10.6%   40.0%     6.4%
n=128,rows=50                   1424     28.48  17.6%  17.6%   17.6%   40.0%    10.5%
n=16,rows=512                   9435     18.43  27.1%  27.1%    3.4%   40.0%     2.0%
n=64,rows=300                   6968     23.23  21.5%  21.5%   10.8%   40.0%     6.5%
n=128,rows=300                  8468     28.23  17.7%  17.7%   17.7%   40.0%    10.6%

ISA aliases: A3 SUB_VV  A4 AGG_RC  A8 MOV_ACC  A9 EXP  A10 FRACTIONAL_SCALAR  A12 ADDRESSING  A13 NARROW_MULT  A14 MULTIPLE_ACC
config                         A3   A4  A8   A9   A10   A12   A13  A14
----------------------------------------------------------------------
n=8,rows=16                    16   16   1   16    32   131    80   19
n=16,rows=16                   16   16   1   16    32   131    80   19
n=32,rows=40                   40   40   1   40    80   339   200   51
n=64,rows=50                   50   50   1   50   100   449   250   76
n=128,rows=50                  50   50   1   50   100   499   250  101
n=16,rows=512                 512  512   4  512  1024  4220  2560  586
n=64,rows=300                 300  300   3  300   600  2697  1500  457
n=128,rows=300                300  300   3  300   600  2997  1500  607
```

mult% / acc%: cycles the stage was busy. lanes%: multiplier lanes that carried data. ident%: busy multiply cycles that were identity multiplies (data routing, not MACs). effMAC%: peak MACs actually retired. ISA aliases: verified occurrences of each alias in docs/content/isa-alias-catalogue.md, all subtypes summed (+N? = unverified candidates); occurrences can overlap, so they are not savings estimates.
