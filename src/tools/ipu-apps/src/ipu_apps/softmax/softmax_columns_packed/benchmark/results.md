# softmax_columns_packed benchmark

```
config                          cycles   cyc/row   mult%    acc%     max_err  status
------------------------------------------------------------------------------------
rows=64,width=8                    234      3.66   24.4%   24.4%   1.490e-07    PASS
rows=64,width=16                   234      3.66   24.4%   24.4%   2.384e-07    PASS
rows=100,width=32                  525      5.25   25.5%   25.5%   1.490e-07    PASS
rows=128,width=64                 1250      9.77   26.0%   26.0%   2.980e-07    PASS
```
