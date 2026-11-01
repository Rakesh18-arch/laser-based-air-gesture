# AIRWRITE V8 — hosted website + cloud API + local Windows bridge

This release keeps the existing desktop application and its camera/gesture processing on the user's Windows PC. Netlify hosts the static website, Render runs the Python account API, Neon stores cloud records, and `server.py` remains a localhost-only desktop bridge.

## Architecture

- **Netlify:** `website/` (HTML, CSS, JavaScript and static assets).
- **Render:** `backend/main.py` (FastAPI account and enquiry API).
- **Neon PostgreSQL:** persistent users, enquiries, activity and sessions.
- **Windows PC:** `START_DESKTOP_BRIDGE.bat` + existing `server.py` + `desktop/` application.

The hosted API deliberately does not expose `/api/desktop/start` or camera status. Those requests go to `http://127.0.0.1:8765` on the visitor's own Windows PC.

## 1. Create the Neon database

1. Create a Neon account and a PostgreSQL project.
2. Copy the pooled or direct PostgreSQL connection string from the Neon dashboard.
3. Keep the connection string private. It is a server-side secret; never put it in `website/app.js`, HTML, or a public GitHub repository.

## 2. Deploy the API to Render

1. Push this release to a private or public GitHub repository (do not commit `.env`, database passwords, or real admin credentials).
2. In Render, create a **Blueprint** from that repository and let it read `render.yaml`.
3. Set these environment variables on the `airwrite-cloud-api` service:
   - `DATABASE_URL` = the Neon PostgreSQL connection string.
   - `AIRWRITE_ADMIN_EMAIL` = the owner's email address.
   - `AIRWRITE_ADMIN_PASSWORD` = a new, unique password (use at least 16 characters).
   - `AIRWRITE_ALLOWED_ORIGINS` = `https://laserhandgesture.netlify.app` (add any custom production domain as a comma-separated extra origin).
4. Deploy. The service health check is `/health`.
5. Copy the Render service hostname, for example `airwrite-cloud-api.onrender.com`.

The first successful startup creates the database tables and the owner account. Do not use the old local `owner` username or the generated `data/OWNER_LOGIN.txt` credentials for the hosted owner account. The hosted owner signs in using `AIRWRITE_ADMIN_EMAIL` and `AIRWRITE_ADMIN_PASSWORD`.

## 3. Connect Netlify to Render

1. Open `website/_redirects`.
2. Replace `YOUR-RENDER-SERVICE` with the exact Render service hostname (without `https://`), for example:
   ` /api/*  https://airwrite-cloud-api.onrender.com/api/:splat  200!`
3. Commit and push the change.
4. In Netlify, deploy the **website** folder as the publish directory. The `_redirects` file must be included in that folder.
5. Open `https://laserhandgesture.netlify.app/health` only if you also add a `/health` proxy rule; otherwise test the Render `/health` URL directly. Test the Netlify `/api/me` route while logged out: it should return HTTP 401 JSON, not a Netlify 404 page.

The `/api/*` Netlify rewrite proxies browser requests to Render while keeping the browser on the Netlify origin. This lets the session cookie remain first-party.

## 4. Start the local Windows desktop bridge

On the Windows computer that will run AIRWRITE:

1. Extract the complete release folder.
2. Double-click `START_DESKTOP_BRIDGE.bat` and keep its terminal window open.
3. Open the hosted website in the browser and sign in.
4. If the browser asks for local-network access to `127.0.0.1`, allow it for the AIRWRITE site. Browser support and permission wording vary.
5. The website should show the desktop bridge as connected. Choose Laser Writer or Hand Gesture Studio.

The local bridge listens only on `127.0.0.1`. Its cross-origin permission is restricted to `AIRWRITE_ALLOWED_ORIGINS` (default: the production Netlify domain). The cloud API never receives camera frames and cannot control a different user's PC.

If using a Netlify deploy-preview URL or custom domain, add that exact HTTPS origin to `AIRWRITE_ALLOWED_ORIGINS` in the Windows bridge environment before launching it. The hosted backend must also include that origin in its own `AIRWRITE_ALLOWED_ORIGINS` environment variable if requests go directly to it; with the included same-origin Netlify proxy, the production Netlify origin is sufficient.

## 5. Important hosting notes

- **Netlify Free:** currently $0 with a 300-credit monthly limit; requests, bandwidth, builds and other usage draw from credits. Check the live plan page before deploying. See [Netlify pricing](https://www.netlify.com/pricing/).
- **Render Free API:** currently spins down after 15 minutes without inbound traffic and takes about one minute to wake on the next request. Render describes free instances as suitable for hobby/testing rather than production. See [Render free-instance limits](https://render.com/docs/free).
- **Neon Free database:** currently includes 1 GB storage and 100 compute-hours per project each month; compute scales to zero after five minutes of inactivity. See [Neon Free-plan quota guide](https://github.com/neondatabase/website/blob/main/content/faqs/free-plan-limits-and-quotas.md).
- These providers can change quotas and eligibility. This is a $0 starter configuration when you stay within the free allowances, not an uptime or production guarantee. Check current account dashboards and terms before a public demo.
- The database is external and persistent; do not switch the hosted service to SQLite on an ephemeral app filesystem.
- Keep admin credentials, `DATABASE_URL`, and any provider tokens out of the frontend and Git history.
- Use HTTPS for the public site. The hosted session cookie is `HttpOnly`, `Secure`, and `SameSite=Lax`.
- The original local `START_AIRWRITE.bat` remains available for the standalone local website + desktop workflow. It uses the original local SQLite database and local owner account. Hosted accounts and local SQLite accounts are separate.

## Troubleshooting

- **Login/register gives 404:** verify `website/_redirects` points to the correct Render hostname and redeploy the `website` folder.
- **API returns 503:** open Render `/health`; check `DATABASE_URL` and Neon availability.
- **Owner dashboard unavailable:** verify `AIRWRITE_ADMIN_EMAIL` and `AIRWRITE_ADMIN_PASSWORD` are set on Render, then inspect Render startup logs.
- **Desktop bridge offline:** run `START_DESKTOP_BRIDGE.bat` on the same Windows PC, keep it open, and allow the browser's local-network request.
- **Bridge CORS error:** add the exact website origin to `AIRWRITE_ALLOWED_ORIGINS` before starting the bridge.
- **App starts locally but hosted site does not see it:** ensure the bridge is listening on `127.0.0.1:8765`; the hosted website and Windows bridge must be used in the same browser on the same PC.
