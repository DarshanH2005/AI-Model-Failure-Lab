# Demo sequence (2 minutes 40 seconds)

The exported captioned video is `artifacts/core-v4/media/demo.mp4`.
It explains the defect signal, correlated background/watermark/lighting, valid interventions, fair repair budgets and actual held-out results.

For a live demonstration, launch `./Start-Lab.ps1 -Mode dashboard`, open Investigation, choose either architecture and edit a development image. Compare original and repaired raw probabilities, inspect masks, then open Repair comparison and Report.

The final 66-run study has a ceiling-limited result: original models already achieve 100% synthetic test accuracy and targeted selection does not improve worst-group accuracy over random nuisance selection. Show this limitation honestly; the development gallery does not establish an accuracy failure. One random-standard repair run makes one error on 1,024 inverted-test images; show that saved regression in the report without using test images to tune the model. Harder natural-image validation requires a separate benchmark and protocol.

