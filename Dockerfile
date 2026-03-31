FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

**Step 2 — Create this file in the same repo**

Filename: `.dockerignore`
```
__pycache__/
*.pyc
*.pyo
.env
.git
.gitignore
*.md
