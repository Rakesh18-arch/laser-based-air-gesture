AIRWRITE STUDIO V8 — START HERE
============================

This release supports two ways to run AIRWRITE:

A) Standalone local website + desktop app
   1. Extract this ZIP.
   2. Double-click START_AIRWRITE.bat.
   3. Register or log in to the local portal.
   4. Choose Start Laser Writer or Start Hand Gesture Studio.
   This mode continues to use the local SQLite database and local owner credentials.

B) Hosted website + cloud accounts + this PC's desktop app
   1. Deploy website/ to Netlify and backend/ to Render, and connect a Neon PostgreSQL database.
   2. Follow DEPLOY_FREE_HOSTING.md.
   3. On the Windows PC that should run the app, double-click START_DESKTOP_BRIDGE.bat.
   4. Keep the bridge window open, open https://laserhandgesture.netlify.app, sign in, and launch a desktop mode.
   The hosted account database is separate from the standalone local SQLite database.

DESKTOP MODES
- Laser Writer retains the existing laser camera profile and calibration workflow.
- Hand Gesture Studio retains the existing hand camera / gesture workflow.
- Camera frames and desktop processing stay on the Windows PC; they are not uploaded to the cloud API.

HOSTED DEPLOYMENT
- Netlify publishes the website/ folder.
- Render runs backend/main.py.
- Neon PostgreSQL stores hosted users, sessions and pilot enquiries.
- Set secrets only in Render environment variables. Never publish DATABASE_URL or the owner password.
- The included website/_redirects file contains a placeholder. Replace YOUR-RENDER-SERVICE with the actual Render hostname before deploying.

See DEPLOY_FREE_HOSTING.md for full setup and troubleshooting.

The original local workflow remains available and has not been removed.
