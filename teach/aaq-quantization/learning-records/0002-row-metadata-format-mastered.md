# Row data, metadata word and table format: understood

The user can place each piece of information about an AaQ row: element bits in the data bytes, `scale` and `dynamic_exponent` in the 16-bit metadata word, `format` in the table configuration, the LUT in the AaQ stage. They also know that decoding needs both the scale and the table's `fe`.

**Evidence**: 8/8 on the lesson 0001 quiz, first try.

**Implications**: next gap was "what is a table, how many per kernel, how does the IPU know it" (the user's own question), taught in lesson 0002. Lesson 0001 terms are now in GLOSSARY.md.
