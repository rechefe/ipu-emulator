# softmax_columns benchmark

```
config                          cycles   cyc/row   mult%    acc%     max_err  status
------------------------------------------------------------------------------------
rows=16,width=128                  327     20.44   24.5%   24.5%   1.490e-07    PASS
rows=64,width=128                 1239     19.36   25.8%   25.8%   3.278e-07    PASS
rows=128,width=128                2455     19.18   26.1%   26.1%   2.682e-07    PASS
rows=32,width=256                 1252     39.12   25.6%   25.6%   2.384e-07    PASS
rows=64,width=384                 3697     57.77   26.0%   26.0%   2.980e-07    PASS
```
