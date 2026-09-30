# Publish to GitHub

## Recommended: GitHub Desktop

1. Open GitHub Desktop and select **File > Add Local Repository**.
2. Choose this directory.
3. If it is not yet a repository, select **Create a repository** and keep the name `utility-tunnel-retroreflective-landmarks`.
4. Review the initial commit. The `.gitignore` excludes bags, point clouds, build folders, and output folders.
5. Select **Publish repository**. Choose **Public** only after confirming that no controlled data, maps, or industrial-site material has been added; otherwise choose **Private**.

Replace `YOUR_ACCOUNT` in `README.md` after the GitHub repository URL is known.

## GitHub CLI

After completing `gh auth login`, run these commands only in this repository root:

```bash
git init
git add .
git commit -m "Release stable reflector observation module"
gh repo create utility-tunnel-retroreflective-landmarks --public --source=. --remote=origin --push
```

Use `--private` instead of `--public` when the repository should not be publicly accessible.

