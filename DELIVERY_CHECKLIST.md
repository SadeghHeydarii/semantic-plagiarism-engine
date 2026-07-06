# چک‌لیست تحویل

- [ ] نام و شماره دانشجویی در `docs/project_spec.tex` تکمیل شود.
- [ ] گزارش با XeLaTeX دوباره ساخته و `docs/project_spec.pdf` بررسی شود.
- [ ] پروژه در GitHub بارگذاری شود و دسترسی استاد/دستیار داده شود.
- [x] کد ماژولار و CLI سه‌دستوری آماده است.
- [x] `outputs/metrics.csv` و `outputs/candidates.csv` تولید شده‌اند.
- [x] آزمون‌ها نوشته و با موفقیت اجرا شده‌اند.
- [x] ویدئوی کوتاه اجرای CLI در `demo/cli_demo.mp4` قرار دارد.
- [x] داده خام بزرگ در Git قرار نمی‌گیرد و `data/raw/` نادیده گرفته می‌شود.

## Version 1.1 calibrated ensemble
- [x] Added 39-feature calibrated lexical ensemble.
- [x] Added deterministic train/validation/test calibration command.
- [x] Added saved-model support to `compare` and candidate scoring in `corpus`.
- [x] Added held-out metrics and predictions for the 15,000-pair Quora subset.
- [x] Added ensemble tests; 20 tests pass.
