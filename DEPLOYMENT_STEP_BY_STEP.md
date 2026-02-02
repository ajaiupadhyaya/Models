# Deploy the Bloomberg Terminal — Step-by-Step Guide

This is the **single, detailed guide** for putting your terminal on the web. Follow one section for your chosen platform. You will: push code to GitHub, connect the repo to a host, set environment variables, and get a live URL where you (and others) can sign in and use the terminal — no need to run anything on your computer.

---

## Before you start

1. **Get API keys** (free):
   - **FRED**: [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) — request an API key, copy it.
   - **Alpha Vantage**: [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) — get your free API key, copy it.
   - **OpenAI** (optional, for AI assistant): [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) — create an API key if you want AI analysis.

2. **Decide sign-in credentials** (what you will type to log in to the terminal):
   - Pick a **username** (e.g. `admin` or your name).
   - Pick a **password** (use a strong one for production).
   - Pick an **AUTH_SECRET**: a long random string (e.g. 32+ characters). You can generate one with: `openssl rand -hex 32` in a terminal, or use a password generator.

3. **GitHub**: Have a GitHub account. Your project should be in a Git repo (local folder with `git init` or already cloned from GitHub).

---

## Part A: Push your code to GitHub

Do this once. If the repo is already on GitHub and up to date, skip to Part B.

### A.1. Create a new repository on GitHub (if you don’t have one)

1. Open [https://github.com](https://github.com) and sign in.
2. Click the **+** (plus) in the top-right → **New repository**.
3. **Repository name**: e.g. `Models` or `bloomberg-terminal`.
4. **Description** (optional): e.g. “Bloomberg-style financial terminal”.
5. Choose **Public**.
6. Do **not** check “Add a README” (you already have code).
7. Click **Create repository**.
8. On the next screen, note the **repository URL** (e.g. `https://github.com/YOUR_USERNAME/Models.git`). You will use this below.

### A.2. Push your local code to GitHub

1. Open a terminal and go to your project folder (where `Dockerfile` and `api/` live):
   ```bash
   cd /path/to/Models
   ```

2. See if Git is already set up:
   ```bash
   git status
   ```
   If you see “not a git repository”, run:
   ```bash
   git init
   ```

3. Add the GitHub repo as `origin` (replace `YOUR_USERNAME` and `Models` with your repo):
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/Models.git
   ```
   If `origin` already exists and you want to replace it:
   ```bash
   git remote set-url origin https://github.com/YOUR_USERNAME/Models.git
   ```

4. Add all files, commit, and push (use the branch name your repo uses, often `main`):
   ```bash
   git add .
   git status
   ```
   Check that `frontend/node_modules` and `frontend/dist` are **not** in the list (they should be in `.gitignore`). If they appear, do not commit them; ensure `.gitignore` contains `frontend/node_modules/` and `frontend/dist/`.

   ```bash
   git commit -m "Unified terminal: auth, SPA serving, Docker multi-stage, deploy docs"
   git branch -M main
   git push -u origin main
   ```

5. If `git push` asks for credentials:
   - **Username**: your GitHub username.
   - **Password**: use a **Personal Access Token** (GitHub no longer accepts account passwords). Create one: GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)** → **Generate new token**. Give it `repo` scope, copy the token, and paste it when prompted for password.

6. After a successful push, refresh your repository page on GitHub. You should see your files (e.g. `api/`, `frontend/`, `Dockerfile`, `DEPLOY.md`).

---

## Part B: Deploy to a host (choose one)

Pick **one** of: Railway, Render, or Fly.io. Each gives you a public URL. Steps are written so you can follow them exactly.

---

## Option 1: Railway (recommended for simplicity)

### B1.1. Sign in to Railway

1. Go to [https://railway.app](https://railway.app).
2. Click **Login** or **Start a New Project**.
3. Choose **Login with GitHub**.
4. Authorize Railway to access your GitHub account (click **Authorize railway-app** or similar).

### B1.2. Create a new project from GitHub

1. On the Railway dashboard, click **New Project**.
2. Select **Deploy from GitHub repo**.
3. If asked, click **Configure GitHub App** and allow Railway access to your repositories (you can choose “All repositories” or only the one you use, e.g. `Models`).
4. In the list of repos, find your repo (e.g. `Models`) and click it.
5. Railway will create a new **service** and start building. You do **not** need to choose a template; Railway will use your **Dockerfile** automatically.

### B1.3. Add environment variables

1. In the project view, click your **service** (the box representing your app).
2. Open the **Variables** tab (or **Settings** → **Variables**).
3. Click **Add Variable** or **New Variable**.
4. Add each of these **one by one** (name exactly as shown, value as described):

   | Variable name         | Where to get the value |
   |-----------------------|-------------------------|
   | `FRED_API_KEY`        | The FRED API key you copied earlier. |
   | `ALPHA_VANTAGE_API_KEY` | The Alpha Vantage API key you copied. |
   | `OPENAI_API_KEY`      | Your OpenAI API key (optional; leave blank if you don’t use AI). |
   | `TERMINAL_USER`        | The username you chose for signing in (e.g. `admin`). |
   | `TERMINAL_PASSWORD`   | The password you chose for signing in. |
   | `AUTH_SECRET`         | The long random string you chose (e.g. output of `openssl rand -hex 32`). |

   Example: Click **Add Variable**, set **Variable** to `FRED_API_KEY`, set **Value** to your key, then save. Repeat for every row.

5. Do **not** add `PORT` — Railway sets it automatically.

### B1.4. Get your public URL

1. Open the **Settings** tab for your service (or the service card).
2. Find **Networking** or **Public Networking**.
3. Click **Generate Domain** (or **Add Domain**). Railway will assign a URL like `https://models-production-xxxx.up.railway.app`.
4. Copy that URL (e.g. `https://models-production-xxxx.up.railway.app`).

### B1.5. Wait for the build and open the app

1. Go to the **Deployments** tab. Wait until the latest deployment shows **Success** or **Active** (first build can take a few minutes).
2. If the status is **Failed**, click the deployment and read the build logs; fix any errors (e.g. typo in Dockerfile, missing file).
3. When the deployment is successful, open the URL you copied in a browser. You should see the **Bloomberg Terminal** sign-in page.
4. Sign in with the **TERMINAL_USER** and **TERMINAL_PASSWORD** you set. You should then see the terminal (watchlist, charts, etc.).

### B1.6. Updating the app later

- Push new commits to the same branch (e.g. `main`) on GitHub. Railway will automatically rebuild and redeploy. No need to run anything on your computer.

---

## Option 2: Render

### B2.1. Sign in to Render

1. Go to [https://render.com](https://render.com).
2. Click **Get Started** or **Sign In**.
3. Choose **Sign in with GitHub**.
4. Authorize Render to access your GitHub account.

### B2.2. Create a Web Service from your repo

1. On the Render dashboard, click **New +** (top right).
2. Select **Web Service**.
3. Under **Connect a repository**, find and click your repository (e.g. `Models`). If it’s not listed, click **Configure account** and grant Render access to the repo, then try again.
4. After selecting the repo, you’ll see the **Create Web Service** form.

### B2.3. Configure the Web Service

1. **Name**: e.g. `bloomberg-terminal` or `models-terminal`. This will appear in the URL (e.g. `bloomberg-terminal.onrender.com`).
2. **Region**: Choose the one closest to you (e.g. Oregon, Frankfurt).
3. **Branch**: Leave as `main` (or the branch you push to).
4. **Root Directory**: Leave blank (your Dockerfile is in the repo root).
5. **Runtime**: Select **Docker**.
6. **Build Command**: Leave blank (Render uses the Dockerfile to build).
7. **Start Command**: Leave blank (Render uses the Dockerfile `CMD`).
8. **Instance Type**: Free is fine to start; you can change it later.

### B2.4. Add environment variables

1. Scroll to **Environment** or **Environment Variables**.
2. Click **Add Environment Variable**.
3. Add each variable **one by one** (key = name, value = your secret):

   | Key                     | Value |
   |-------------------------|--------|
   | `FRED_API_KEY`          | Your FRED API key. |
   | `ALPHA_VANTAGE_API_KEY` | Your Alpha Vantage API key. |
   | `OPENAI_API_KEY`        | Your OpenAI API key (optional). |
   | `TERMINAL_USER`         | Your chosen login username. |
   | `TERMINAL_PASSWORD`     | Your chosen login password. |
   | `AUTH_SECRET`           | Your long random secret string. |

   Do **not** set `PORT` — Render sets it.

### B2.5. Create the service and get the URL

1. Click **Create Web Service** at the bottom.
2. Render will start building (you’ll see build logs). The first build can take several minutes (Node builds the frontend, then Python image is built).
3. When the build finishes successfully, the **Logs** will show the app running. At the top of the page you’ll see **Your service is live at** followed by a URL like `https://bloomberg-terminal.onrender.com`.
4. Copy that URL and open it in a browser. You should see the sign-in page; sign in with your credentials.

### B2.6. Free tier note

- On the free tier, the service may **spin down** after inactivity. The first request after that can take 30–60 seconds (cold start). For always-on, use a paid plan.

### B2.7. Updating the app later

- Push to `main` (or the branch you selected). Render will automatically redeploy.

---

## Option 3: Fly.io

### B3.1. Install Fly CLI and log in

1. Install the Fly CLI:
   - **macOS (Homebrew)**: `brew install flyctl`
   - **Windows (PowerShell)**: `powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"`
   - **Linux**: `curl -L https://fly.io/install.sh | sh`
   - Or see: [https://fly.io/docs/hub/installing-flyctl/](https://fly.io/docs/hub/installing-flyctl/)

2. Log in:
   ```bash
   fly auth login
   ```
   Follow the prompts (browser will open to sign in with Fly or GitHub).

### B3.2. Launch the app from your repo

1. In a terminal, go to your project root (where `Dockerfile` is):
   ```bash
   cd /path/to/Models
   ```

2. Run:
   ```bash
   fly launch
   ```

3. When prompted:
   - **App name**: Accept the suggestion or type one (e.g. `models-terminal`). It must be unique on Fly.
   - **Region**: Pick the one closest to you (e.g. `iad` for Virginia).
   - **Postgres / Redis**: Say **No** (we don’t need them for the basic terminal).
   - Fly may ask to create a `fly.toml`; say **Yes**.

4. Do **not** deploy yet when it asks; we’ll set secrets first. If it deployed already, that’s OK — we’ll set secrets and redeploy.

### B3.3. Set secrets (environment variables)

Run these in the same folder, replacing the placeholder values with your real ones:

```bash
fly secrets set FRED_API_KEY=your_fred_key_here
fly secrets set ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
fly secrets set OPENAI_API_KEY=your_openai_key_here
fly secrets set TERMINAL_USER=admin
fly secrets set TERMINAL_PASSWORD=your_secure_password
fly secrets set AUTH_SECRET=your_long_random_secret_string
```

Use your actual keys and credentials. Each command will confirm the secret was set.

### B3.4. Deploy

```bash
fly deploy
```

Fly will build the image (using your Dockerfile) and deploy. When it finishes, it will print the app URL (e.g. `https://models-terminal.fly.dev`).

### B3.5. Open the app

1. Open the URL in a browser (or run `fly open`).
2. You should see the sign-in page. Sign in with `TERMINAL_USER` and `TERMINAL_PASSWORD`.

### B3.6. Updating the app later

From the project root:

```bash
fly deploy
```

To change secrets later: `fly secrets set VARIABLE_NAME=new_value`, then `fly deploy` if needed.

---

## Part C: After deployment (all platforms)

1. **Bookmark your URL** so you can open the terminal anytime.
2. **Share the URL** (and login credentials if you want) with others; they can use it from any browser without installing anything.
3. **Changing credentials**: Update `TERMINAL_USER`, `TERMINAL_PASSWORD`, or `AUTH_SECRET` in the host’s environment (Railway Variables, Render Environment, or `fly secrets set`), then redeploy if the platform doesn’t auto-reload.
4. **API docs**: You can open `https://YOUR_URL/docs` to see the FastAPI docs (optional).

---

## Quick reference: environment variables

| Variable                 | Required | Where to get it |
|--------------------------|----------|------------------|
| `FRED_API_KEY`           | Yes      | [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `ALPHA_VANTAGE_API_KEY`  | Yes      | [Alpha Vantage](https://www.alphavantage.co/support/#api-key) |
| `OPENAI_API_KEY`         | No       | [OpenAI](https://platform.openai.com/api-keys) (for AI assistant) |
| `TERMINAL_USER`          | Yes      | Your chosen login username |
| `TERMINAL_PASSWORD`      | Yes      | Your chosen login password |
| `AUTH_SECRET`            | Yes      | Long random string (e.g. `openssl rand -hex 32`) |
| `PORT`                   | No       | Set by the host; do not set manually |

---

## Troubleshooting

- **Build fails**: Check the host’s build logs. Common issues: typo in Dockerfile, missing file (e.g. `frontend/package.json`), or branch not pushed (e.g. still on `main`).
- **“Invalid username or password”**: Ensure `TERMINAL_USER` and `TERMINAL_PASSWORD` are set exactly (no extra spaces) on the host.
- **Blank or 502 page**: Wait a minute after deploy (cold start on free tiers). If it persists, check the service logs on the host.
- **Charts/data not loading**: Confirm `FRED_API_KEY` and `ALPHA_VANTAGE_API_KEY` are set and valid; check the browser Network tab and the host logs for API errors.

For more context, see [DEPLOY.md](DEPLOY.md) and [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md).
