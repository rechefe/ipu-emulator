# softmax_rows_long benchmark

```
config                        cycles  cyc/rows  mult%   acc%  lanes%  ident%  effMAC%
-------------------------------------------------------------------------------------
rows=8,n=200                     402     50.25  19.9%  19.9%   18.2%   40.0%    11.1%
rows=8,n=300                     570     71.25  21.1%  21.1%   19.2%   40.0%    11.7%
rows=16,n=500                   1458     91.12  21.9%  21.9%   21.7%   40.0%    13.1%
rows=4,n=1000                    714    178.50  22.4%  22.4%   22.2%   40.0%    13.3%
rows=200,n=200                  9634     48.17  20.8%  20.8%   18.9%   40.0%    11.5%
rows=300,n=129                 14450     48.17  20.8%  20.8%   16.6%   40.0%    10.4%

ISA aliases: A3 SUB_VV  A4 AGG_RC  A8 MOV_ACC  A9 EXP  A10 FRACTIONAL_SCALAR  A12 ADDRESSING  A13 NARROW_MULT  A14 MULTIPLE_ACC
config                         A3   A4  A8   A9   A10   A12   A13   A14
-----------------------------------------------------------------------
rows=8,n=200                   16   16   1   16    32   114    80    33
rows=8,n=300                   24   24   1   24    48   178   120    49
rows=16,n=500                  64   64   1   64   128   482   320   129
rows=4,n=1000                  32   32   1   32    64   250   160    65
rows=200,n=200                400  400   2  400   800  2804  2000   803
rows=300,n=129                600  600   3  600  1200  4206  3000  1205
```

mult% / acc%: cycles the stage was busy. lanes%: multiplier lanes that carried data. ident%: busy multiply cycles that were identity multiplies (data routing, not MACs). effMAC%: peak MACs actually retired. ISA aliases: verified occurrences of each alias in docs/content/isa-alias-catalogue.md, all subtypes summed (+N? = unverified candidates); occurrences can overlap, so they are not savings estimates.
