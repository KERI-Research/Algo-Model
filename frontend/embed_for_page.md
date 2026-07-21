# Embed KERI Dashboard Into keri.ceberallab.org (GitHub Pages)

This guide makes the React dashboard embeddable inside your existing website.

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

## 3) Build for `/dashboard` path

```bash
cd frontend
npm run build:pages
```

This produces a static build with asset paths rooted at `/dashboard`.

## 4) Publish build to GitHub Pages

If this frontend repo is the one connected to Pages:

```bash
cd frontend
npm run deploy
```

If your website is in a different repo, copy the generated `frontend/build` contents to the `dashboard/` folder of the website repo and publish that repo.

## 5) Add iframe to website page

Use this HTML snippet in the site where you want the dashboard to appear:

```html
<section class="keri-dashboard-block">
  <h2>KERI Model Dashboard</h2>
  <p>Interactive causal and biomarker analysis interface.</p>

  <iframe
    src="https://keri.ceberallab.org/dashboard/"
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
- If you later lock down CORS, include `https://keri.ceberallab.org` in the allowlist.
- If you choose a different subpath than `/dashboard`, update `build:pages` and iframe `src` together.
