# maxpool2d_stride2 benchmark

```
config                            cycles  mult%   acc%  lanes%  ident%  effMAC%
-------------------------------------------------------------------------------
channels=1,height=2,width=256         29  34.5%  34.5%   34.5%  100.0%     0.0%
channels=64,height=2,width=256      1604  39.9%  39.9%   39.9%  100.0%     0.0%

ISA aliases: A1 MOV_RC  A4 AGG_RC  A8 MOV_ACC  A12 ADDRESSING  A13 NARROW_MULT
config                             A1   A4   A8  A12  A13
---------------------------------------------------------
channels=1,height=2,width=256       2    8    2    6   10
channels=64,height=2,width=256    128  512  128  384  640
```

mult% / acc%: cycles the stage was busy. lanes%: multiplier lanes that carried data. ident%: busy multiply cycles that were identity multiplies (data routing, not MACs). effMAC%: peak MACs actually retired. ISA aliases: verified occurrences of each alias in docs/content/isa-alias-catalogue.md, all subtypes summed (+N? = unverified candidates); occurrences can overlap, so they are not savings estimates.
