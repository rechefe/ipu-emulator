# Notes

- The user asked for technical-writing style: facts and steps, active voice, no filler.
- The user follows along in the claude.ai web app. Publish the workspace as one artifact (home page + lessons + reference + assets) and send the link.
- The workspace lives in `teach/aaq-quantization/` in the repo, uncommitted until the user says otherwise.
- The teaching runs alongside a grilling session on the emulator design. Open design questions Q7–Q12 wait until the user feels ready.
- Colour code used in every lesson: blue = row data bytes, amber = row metadata word, violet = table configuration, teal = AaQ stage.
- The user asked for all remaining lessons (4–8) at once rather than one per session. Lessons 5–8 were drafted in parallel and checked for quiz consistency, links, script errors and phone width.
- Lesson 8 corrected an earlier recommendation: INT8 goldens depend on the 512-byte ST.AAQ (vector_tour, proj_p4, test_execute.py), so Q8 now keeps 512 bytes for legacy data.
