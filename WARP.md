# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository Purpose

This is a Git learning repository designed for practicing Git commands and understanding version control workflows. The repository contains minimal files to focus on Git operations rather than complex code management.

## Common Git Commands

### Basic Repository Operations
```bash
# Check repository status
git status

# View commit history
git --no-pager log --oneline

# View detailed commit history
git --no-pager log --graph --pretty=format:'%h -%d %s (%cr) <%an>' --abbrev-commit

# Show differences in working directory
git diff

# Show differences between staged and last commit
git diff --cached
```

### File Operations
```bash
# Stage files for commit
git add <filename>
git add .  # stage all changes

# Commit changes
git commit -m "commit message"

# Restore files from working directory changes
git restore <filename>

# Unstage files
git restore --staged <filename>
```

### Branch Operations
```bash
# List all branches
git branch -a

# Create and switch to new branch
git checkout -b <branch-name>
git switch -c <branch-name>  # modern alternative

# Switch between branches
git checkout <branch-name>
git switch <branch-name>  # modern alternative

# Merge branches
git merge <branch-name>

# Delete branch
git branch -d <branch-name>
```

### Remote Operations
```bash
# Add remote repository
git remote add origin <url>

# View remotes
git remote -v

# Push to remote
git push origin <branch-name>

# Pull from remote
git pull origin <branch-name>

# Clone repository
git clone <url>
```

## Repository Structure

The repository follows a minimal structure:
- `readme.txt` - Simple text file for practicing file operations
- `.git/` - Git metadata directory (managed by Git)

## Learning Workflow

This repository is ideal for:
1. **Basic Git operations** - Adding, committing, and managing file changes
2. **Branch management** - Creating, switching, and merging branches
3. **History exploration** - Understanding commit history and diffs
4. **Remote operations** - When connected to a remote repository
5. **Conflict resolution** - Practicing merge conflict resolution

## Configuration Notes

- Repository is configured with user "Townlin" <Nonoxiaogao@outlook.com>
- Uses macOS keychain for credential management
- Git LFS is enabled for large file support
- Case-insensitive filesystem support enabled

## Current State

- Single `master` branch with initial commit
- `readme.txt` has uncommitted changes
- No remote repository configured
- `.DS_Store` file is untracked (should be gitignored for macOS)

## Suggested .gitignore

Consider adding a `.gitignore` file with common exclusions:
```
.DS_Store
*.log
*.tmp
*~
```

This repository serves as a safe environment for learning Git fundamentals without the complexity of a full codebase.
