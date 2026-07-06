# قرار دادن پروژه در GitHub

پس از ساخت یک مخزن خالی در حساب GitHub، در PowerShell و داخل پوشه پروژه اجرا کنید:

```powershell
git init
git add .
git commit -m "Complete semantic plagiarism engine project"
git branch -M main
git remote add origin https://github.com/USERNAME/semantic-plagiarism-engine.git
git push -u origin main
```

سپس از قسمت **Settings > Collaborators** دسترسی استاد و دستیار آموزشی را اضافه کنید.

پیش از تحویل، نام و شماره دانشجویی را در `docs/project_spec.tex` وارد کنید و فایل PDF را دوباره با XeLaTeX بسازید.
