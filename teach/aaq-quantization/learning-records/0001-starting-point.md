# Starting point: rows carry metadata; unsure where the format lives

The user knew that every 128-element XMEM row gets metadata beside it and that elements can be INT8 or an FP8 ExMy variant. Their open question was whether the metadata also records the row's format, or whether the format is configured elsewhere and constant across rows. That question blocked every downstream design decision (Q7–Q12).

**Evidence**: the user's own question in the grilling session; answered in chat (format is a table property; metadata holds `scale` and `dynamic_exponent`). Not yet tested.

**Implications**: lesson 0001 tests this distinction with retrieval practice before moving on to the encoding itself.
