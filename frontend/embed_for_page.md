# Embed KERI Dashboard Into keri.cerebrallab.org (GitHub Pages)

This guide keeps one GitHub Pages site while still letting the React dashboard be developed and built separately.

## 1) Install frontend dependencies

```bash
cd frontend
npm install
```

## 2) Set production API URL

Point the frontend to your deployed backend API before building.

```bash
cd frontend
export REACT_APP_API_URL=https://your-api-domain.example
```

Use a real public API URL reachable from browsers.

## 3) Build for `/dashboard` path and sync into org site

```bash
cd frontend
npm run publish:org-site
```

This does two things:

- Builds the React app with asset paths rooted at `/dashboard`.
- Copies the build output into `Org-Site/Keri-Project/dashboard/`.

## 4) Publish only the org site repository to GitHub Pages

Commit and push changes from `Org-Site/Keri-Project` (including the updated `dashboard/` folder). Keep GitHub Pages pointed at the org site repo only.

## 5) Add iframe to website page

Use this HTML snippet in the site where you want the dashboard to appear:

```html
<section class="keri-dashboard-block">
  <h2>KERI Model Dashboard</h2>
  <p>Interactive causal and biomarker analysis interface.</p>

  <iframe
    src="/dashboard/"
    title="KERI Dashboard"
    loading="lazy"
    referrerpolicy="strict-origin-when-cross-origin"
    style="width:100%;min-height:1100px;border:1px solid #d0d7de;border-radius:12px;background:#fff;"
  ></iframe>
</section>
```

## 6) Optional responsive container styling

```css
.keri-dashboard-block {
  max-width: 1200px;
  margin: 0 auto;
  padding: 1rem;
}

.keri-dashboard-block iframe {
  display: block;
}
```

## Notes

- The backend currently allows all origins, so the iframe can call the API from your site domain.
- If you later lock down CORS, include `https://keri.cerebrallab.org` in the allowlist.
- If you choose a different subpath than `/dashboard`, update `build:pages` and iframe `src` together.
