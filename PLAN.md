I want to add a new instruction 

`acc.max` (should also receive `acc.max.first` - same logic as the logic in `acc.first` - ignore old racc values and treat them as 0)

One argument - aaq_reg - AaqRegField

For each element i in the product result and in the accumulation reg - 

racc[i] = max(racc[i], mult_res[i], aaq_reg[aaq_reg])

If its first - racc is excluded for the maximum search