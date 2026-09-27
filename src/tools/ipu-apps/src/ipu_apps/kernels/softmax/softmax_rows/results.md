# softmax_rows benchmark

```
config                        cycles  cyc/rows  mult%   acc%  lanes%  ident%  effMAC%
-------------------------------------------------------------------------------------
rows=8                           162     20.25  24.7%  24.7%   24.7%   40.0%    14.8%
rows=32                          594     18.56  26.9%  26.9%   26.9%   40.0%    16.2%
rows=128                        2323     18.15  27.6%  27.6%   27.6%   40.0%    16.5%
rows=256                        4643     18.14  27.6%  27.6%   27.6%   40.0%    16.5%
rows=500                        9066     18.13  27.6%  27.6%   27.6%   40.0%    16.5%

ISA aliases: A3 SUB_VV  A4 AGG_RC  A8 MOV_ACC  A9 EXP  A10 FRACTIONAL_SCALAR  A12 ADDRESSING  A13 NARROW_MULT  A14 MULTIPLE_ACC
config                         A3   A4  A8   A9   A10   A12   A13   A14
-----------------------------------------------------------------------
rows=8                          8    8   1    8    16    78    40    17
rows=32                        32   32   1   32    64   318   160    65
rows=128                      128  128   1  128   256  1278   640   257
rows=256                      256  256   2  256   512  2556  1280   515
rows=500                      500  500   4  500  1000  4992  2500  1007
```

mult% / acc%: cycles the stage was busy. lanes%: multiplier lanes that carried data. ident%: busy multiply cycles that were identity multiplies (data routing, not MACs). effMAC%: peak MACs actually retired. ISA aliases: verified occurrences of each alias in docs/content/isa-alias-catalogue.md, all subtypes summed (+N? = unverified candidates); occurrences can overlap, so they are not savings estimates.
