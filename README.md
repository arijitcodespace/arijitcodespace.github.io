# GitHub Pages Starter (Publications + Blog)

A minimal Jekyll site using the `minima` theme. It includes:
- A **Publications** page (`/publications/`).
- A **Blog** tab that lists posts from `_posts/`.
- A home page that also shows recent posts.

## Quick start

1. **Create the repo**  
   Make a new repo named `<your-username>.github.io` on GitHub.

2. **Upload these files**  
   Upload everything in this folder to the root of that repo (or push with git).

3. **Enable GitHub Pages**  
   - Go to **Settings → Pages**.
   - Under **Build and deployment**, pick **Deploy from a branch**.
   - Set **Branch** to `main` and **Folder** to `/ (root)`.
   - Save. After it builds, your site will be at `https://<your-username>.github.io/`.

   > Alternative: choose **GitHub Actions** and use the "Jekyll" workflow template if you prefer Actions-based builds.

4. **Edit nav / site text**  
   - `_config.yml` controls the site title and navigation.
   - `publications.md` is where you add your papers.
   - Add new blog posts in `_posts/` named like `YYYY-MM-DD-title.md` with front matter.

## Custom domain (optional)
Add your domain in **Settings → Pages**; then create a `CNAME` record at your DNS provider pointing to `<your-username>.github.io`.

## Local preview (optional)
If you want to run locally:
```bash
bundle install
bundle exec jekyll serve
```
Then open `http://localhost:4000`.