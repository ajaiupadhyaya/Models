# Fix System Python on macOS (for all projects)

If you see **externally-managed-environment** when running `pip install` (even inside a venv), Homebrew’s Python is your default and it blocks pip. Fix it **once** by making a different Python your default. After that, the standard workflow works everywhere:

- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`

---

## Recommended: Use pyenv as your system Python

This makes **pyenv’s Python** your default `python3` in every terminal. Venvs you create will use that Python as their base, so pip works normally in all projects.

### 1. Install pyenv

```bash
brew install pyenv
```

### 2. Install a Python version (e.g. 3.11)

```bash
pyenv install 3.11.0
```

### 3. Set it as your global default

```bash
pyenv global 3.11.0
```

### 4. Wire pyenv into your shell

Add this to your shell config so every new terminal uses pyenv’s Python:

**If you use Bash** (e.g. `~/.bash_profile` or `~/.bashrc`):

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

**If you use Zsh** (e.g. `~/.zshrc`):

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Then reload your shell:

```bash
source ~/.zshrc   # or source ~/.bash_profile
```

### 5. Confirm

```bash
which python3
# Should show something like: /Users/you/.pyenv/shims/python3

python3 --version
# Should show: Python 3.11.0 (or whatever you installed)
```

### 6. Use the standard workflow everywhere

In **any** project:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No more externally-managed-environment. Homebrew’s Python is still installed but your shell uses pyenv’s Python by default.

---

## Alternative: Use Python from python.org

If you prefer not to use pyenv:

1. Download the macOS installer from [python.org/downloads](https://www.python.org/downloads/).
2. Run the installer (it puts Python in `/Library/Frameworks/Python.framework/` or under your user).
3. Put that Python **before** Homebrew in your PATH.

In `~/.zshrc` or `~/.bash_profile`:

```bash
# Use python.org Python first (adjust 3.12 to your version)
export PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin:$PATH"
```

Reload your shell, then check:

```bash
which python3
# Should be under /Library/Frameworks/Python.framework/...
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## After the fix

- **All projects:** Use the normal flow: `python3 -m venv venv`, `source venv/bin/activate`, `pip install -r requirements.txt`.
- **This project:** Follow [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md) “How to launch” with no special Python steps.

You only need to do this once per machine.
