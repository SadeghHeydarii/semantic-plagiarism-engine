# بهبود مدل تشخیص شباهت — نسخه 1.1

در این نسخه، دو مسیر اصلی پروژه یعنی `MinHash + LSH` و `TF-IDF SimHash` حذف یا جایگزین نشده‌اند. یک مسیر تکمیلی و قابل توضیح با نام **Calibrated Lexical Ensemble** اضافه شده است.

## چه چیزی اضافه شده است؟

- استخراج ۱۵ ویژگی پایه از هر جفت متن:
  - شباهت TF-IDF کلمه‌ای
  - شباهت TF-IDF کلمه‌ای همراه با bigram
  - شباهت TF-IDF کاراکتری ۳ تا ۵ گرمی
  - Jaccard توکن‌ها
  - ضریب هم‌پوشانی
  - نسبت طول دو متن
  - تطابق یا تفاوت کلمه پرسشی
  - تفاوت منفی‌سازها
  - شباهت ترتیب کاراکترها
  - شباهت توکن‌های مرتب‌شده
  - Jaccard بیگرام‌ها
  - یکسان بودن اولین توکن
  - نسبت پیشوند مشترک
  - یکسان بودن دقیق دو متن
- ساخت ویژگی‌های درجه دوم و تعاملی؛ در مجموع ۳۹ ویژگی.
- پیاده‌سازی رگرسیون لجستیک در خود پروژه با روش Newton/IRLS و NumPy؛ از `scikit-learn` استفاده نشده است.
- تقسیم قطعی و طبقه‌بندی‌شده داده‌ها به Train / Validation / Test.
- محاسبه IDF فقط روی Train و انتخاب Threshold فقط روی Validation.
- ذخیره مدل کالیبره‌شده در JSON و امکان استفاده از آن در دستورات `compare` و `corpus`.

## اجرای آموزش و ارزیابی

```powershell
python -m plagiarism_engine.cli calibrate `
  --pairs data/raw/quora/train.csv `
  --text-col-a question1 `
  --text-col-b question2 `
  --label-col is_duplicate `
  --id-col id `
  --adaptive-short-text `
  --output outputs/calibrated_metrics.csv `
  --predictions-output outputs/calibrated_predictions.csv `
  --model-output outputs/calibrated_ensemble.json
```

## نتیجه روی بخش Test نگه‌داشته‌شده

| روش | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| Character TF-IDF cosine | 0.509 | 0.821 | 0.628 | 0.638 |
| Token Jaccard | 0.503 | 0.877 | 0.639 | 0.631 |
| Calibrated lexical ensemble | 0.543 | 0.848 | 0.662 | 0.677 |

این اعداد روی داده‌های Test محاسبه شده‌اند و نمونه‌های Test در آموزش وزن‌ها یا IDF استفاده نشده‌اند.

## استفاده از مدل ذخیره‌شده

```powershell
python -m plagiarism_engine.cli compare `
  --file-a data/sample_corpus/doc_01.txt `
  --file-b data/sample_corpus/doc_02.txt `
  --ensemble-model outputs/calibrated_ensemble.json `
  --output outputs/two_file_compare_ensemble.json
```

```powershell
python -m plagiarism_engine.cli corpus `
  --data data/sample_corpus `
  --ensemble-model outputs/calibrated_ensemble.json `
  --output outputs/candidates_ensemble.csv
```

## تست‌ها

هر ۲۰ تست پروژه با موفقیت اجرا شده‌اند.
