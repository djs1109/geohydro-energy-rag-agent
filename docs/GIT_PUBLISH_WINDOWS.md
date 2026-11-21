# Windows PowerShell 发布步骤

在 `D:\Duan\agent_project\agent-learning` 下执行。

```powershell
git --version
if (-not (Test-Path .git)) { git init }

# 先复制好 .gitignore、.env.example、README.md、docs、tools
git status --short

git add .gitignore .env.example README.md docs tools
git add -- *.py requirements*.txt Dockerfile* compose.yaml
if (Test-Path pages) { git add pages }
if (Test-Path tests) { git add tests }

# 避免把密钥或数据库推上去
git diff --cached --name-only
python tools/preflight.py
git diff --cached

git commit -m "feat: professional-domain RAG agent"
git branch -M main

# GitHub 网页先创建一个空仓库，再执行以下两条（替换自己的地址）
# git remote add origin https://github.com/<username>/<repo>.git
# git push -u origin main
```

**不要执行** `docker compose down -v`、`git clean -fdx`、`git push --force`。

若曾经把密钥提交到历史记录，`.gitignore` 无法修复历史泄露，需要立即轮换密钥并审查历史。
