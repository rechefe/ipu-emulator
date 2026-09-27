# maxpool2d_stride2_tail benchmark

```
config                              cycles  mult%   acc%  lanes%  ident%  effMAC%
---------------------------------------------------------------------------------
channels=1,height=2,width=260           41  36.6%  36.6%   36.6%  100.0%     0.0%
channels=64,height=2,width=260        2309  41.6%  41.6%   41.6%  100.0%     0.0%
channels=1,height=480,width=640      12488  48.0%  48.0%   48.0%  100.0%     0.0%
channels=64,height=480,width=640    798917  48.1%  48.1%   48.1%  100.0%     0.0%

ISA aliases: A1 MOV_RC  A4 AGG_RC  A8 MOV_ACC  A12 ADDRESSING  A13 NARROW_MULT
config                                 A1      A4     A8     A12     A13
------------------------------------------------------------------------
channels=1,height=2,width=260           3      12      3       9      15
channels=64,height=2,width=260        192     768    192     576     960
channels=1,height=480,width=640      1200    4800   1200    3600    6000
channels=64,height=480,width=640    76800  307200  76800  230400  384000
```

mult% / acc%: cycles the stage was busy. lanes%: multiplier lanes that carried data. ident%: busy multiply cycles that were identity multiplies (data routing, not MACs). effMAC%: peak MACs actually retired. ISA aliases: verified occurrences of each alias in docs/content/isa-alias-catalogue.md, all subtypes summed (+N? = unverified candidates); occurrences can overlap, so they are not savings estimates.
