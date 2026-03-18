# 本地代码自动同步到 GitHub（Windows PowerShell）

本文档对应仓库内两个脚本：

- `scripts/github_bootstrap.ps1`：首次初始化，自动创建 GitHub 仓库并首推
- `scripts/github_sync.ps1`：后续日常提交并推送

## 1. 准备 GitHub Token

1. 在 GitHub 创建 Personal Access Token（建议 `classic`，至少包含 `repo` 权限）。
2. 在当前 PowerShell 会话设置（立即生效）：

```powershell
$env:GITHUB_TOKEN = "你的Token"
```

3. 可选：写入用户环境变量（新终端生效）：

```powershell
setx GITHUB_TOKEN "你的Token"
```

## 2. 首次初始化并创建远程仓库

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\github_bootstrap.ps1 `
  -RepoName "time-llm-your-project" `
  -Owner "你的GitHub用户名或组织名" `
  -Visibility private `
  -Description "Time-LLM customized project"
```

说明：

- 如果本地还不是 git 仓库，脚本会自动 `git init`。
- 脚本会自动创建 `origin` 远端并首推到 `main`。
- 若远端仓库已存在，会自动复用并更新远端地址。

## 3. 日常自动提交并推送

每次改完代码后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\github_sync.ps1 `
  -Message "feat: 描述本次修改"
```

脚本会自动执行 `git add -A`、`commit`、`push`。

## 4. 安全与体积保护

脚本内置了以下保护：

- 检测暂存区大文件（>100MB）并阻止 push，避免 GitHub 上传失败
- 默认不在命令行明文拼接远程 URL Token

另外，项目已配置 `.gitignore`，默认忽略了：

- `pretrained_models/`
- `scripts/checkpoints/`
- `visual*` 可视化产物
- 以及常见缓存/日志文件

## 5. 常用参数

`github_bootstrap.ps1`：

- `-RepoName`：仓库名（必填）
- `-Owner`：用户名/组织名（默认自动读取当前 Token 对应用户）
- `-Visibility`：`private` 或 `public`
- `-SkipInitialCommit`：跳过自动初始提交
- `-NoPush`：只配置，不推送

`github_sync.ps1`：

- `-Message`：提交信息
- `-Branch`：目标分支（默认当前分支）
- `-NoPush`：只提交，不推送
- `-AllowEmpty`：允许空提交

