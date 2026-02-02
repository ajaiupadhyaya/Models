# Render CLI – Build Logs & Troubleshooting

Use the [Render CLI](https://render.com/docs/cli) locally so you can share build logs and fix deploy failures without leaving the dashboard.

## 1. Install (one-time)

**macOS (Homebrew):**
```bash
brew install render
```

**Linux / other:** See [Render CLI docs](https://render.com/docs/cli) (curl script or GitHub releases).

Confirm:
```bash
render --version
```

## 2. Log in (one-time per machine)

```bash
render login
```

- Opens your browser to the Render Dashboard.
- Click **Generate token**; the CLI saves it locally.
- When prompted, choose the **workspace** that contains your service (e.g. `terminal-api`).

## 3. Get your service ID

List services in the active workspace:

```bash
render services -o json --confirm
```

Find your web service (e.g. `terminal-api`) and note its **id** (e.g. `srv-xxxx`). You can also copy the service ID from the Render Dashboard URL:  
`https://dashboard.render.com/web/srv-xxxx` → `srv-xxxx`.

## 4. View deploy history and build logs

**List recent deploys (interactive – pick one to open logs):**
```bash
render deploys list <SERVICE_ID>
```

Example:
```bash
render deploys list srv-abc123xyz
```

- In interactive mode you can select a deploy to view its **build and runtime logs**.
- Use this to inspect the **failed** deploy and copy the build log output.

**List deploys as JSON (for scripting or to find deploy IDs):**
```bash
render deploys list <SERVICE_ID> -o json --confirm
```

## 5. Stream logs (service or deploy)

**Live service logs:**
```bash
render logs -r <SERVICE_ID> --tail
```

**Recent logs (no stream), e.g. last 200 lines:**
```bash
render logs -r <SERVICE_ID> --limit 200 -o text --confirm
```

Build logs are shown when you select a deploy from `render deploys list <SERVICE_ID>` in interactive mode.

## 6. Trigger a new deploy (e.g. after a fix)

```bash
render deploys create <SERVICE_ID>
```

- By default this triggers a deploy and **tails logs** (build + runtime).
- To wait until the deploy finishes and get a non-zero exit code on failure:
  ```bash
  render deploys create <SERVICE_ID> --wait
  ```
- To clear build cache before deploying:
  ```bash
  render deploys create <SERVICE_ID> --clear-cache --wait
  ```

## 7. Validate `render.yaml` (Blueprint)

From the repo root:

```bash
render blueprints validate ./render.yaml
```

Use this before pushing if you change the Blueprint.

---

## Quick reference for “build failed” debugging

1. **Install & log in:** `brew install render` then `render login`.
2. **Get service ID:** `render services -o json --confirm` (or from dashboard URL).
3. **Open failed build logs:**  
   `render deploys list <SERVICE_ID>` → select the failed deploy → view logs.
4. **Copy the build log** (especially the last 50–100 lines around the error) and share it so we can fix the Dockerfile, dependencies, or config.
5. **After fixing:** push to your branch; Render will auto-deploy, or run  
   `render deploys create <SERVICE_ID> --wait` to deploy and wait.

For full CLI options: `render help` and `render <command> --help`.
