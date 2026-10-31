# arijitcodespace.github.io

Personal website of Arijit Dey, built with [Jekyll](https://jekyllrb.com/).
Every push to `main` is built and deployed to GitHub Pages by GitHub Actions
(`.github/workflows/pages.yml`); progress and errors show up in the repo's **Actions** tab.

## Writing a blog post

1. Copy `_drafts/paper-explainer-template.md` to `_posts/YYYY-MM-DD-short-name.md`
   (for example `_posts/2026-11-02-attention-is-all-you-need.md`).
2. Edit the front matter: `title`, and `description` (the summary shown on the blog page).
3. Put the post's images in `assets/blog/short-name/`.
4. Add BibTeX entries for the papers you cite to `_bibliography/references.bib`.
5. Commit and push. The post is live on `/blog/` a minute or two later.

Posts are written in Markdown with LaTeX for math, citations and cross-references:

| You write | You get |
| --- | --- |
| `$x_i$` or `\(x_i\)` | inline math |
| `$$ ... $$` on its own lines, or `\[ ... \]` | displayed equation |
| `\begin{equation} ... \label{eq:loss} \end{equation}` (also `align`, `gather`, `multline`) | numbered equation |
| `\eqref{eq:loss}` / `\ref{eq:loss}` | clickable "(1)" / "1" |
| `{% include figure.html src="plot.png" label="fig:plot" caption="..." %}` | numbered figure with caption |
| `Figure~\ref{fig:plot}` | "Figure 2", linked to the figure |
| `\cite{key}`, `\cite{a,b}`, `\cite[Sec.~3]{key}` | [1], [1, 2], [1, Sec. 3], linked to the reference list |
| `\citet{key}` | Vaswani et al. [1] |
| `text[^note]` + `[^note]: ...` | footnote |
| `\$` | a literal dollar sign |

The **References** list is generated at the end of each post from the entries it cites,
numbered in order of first citation, with each title linked to the paper (URL, DOI or arXiv ID).
Figure captions can contain Markdown, math and citations. For a caption with double quotes in it,
use single quotes around it: `caption='...'`.

LaTeX macros: add them per post with `macros:` in the front matter, or for every post with
`math_macros:` in `_config.yml`:

```yaml
macros:
  KL: ['D_{\mathrm{KL}}\left(#1 \,\|\, #2\right)', 2]   # \KL{p}{q}, two arguments
  x: '\mathbf{x}'
```

Good to know:

- Math and citation commands are left alone inside fenced code blocks (```` ``` ````) and `inline code`.
- `$5 and $10` stays as text: a closing `$` followed by a digit is never treated as math.
- The build log in the Actions tab warns about a missing BibTeX key or figure label, and the post
  shows a red `?` at that spot.
- To switch the LaTeX handling off for one post (say, one full of dollar amounts), add `math: false`
  to its front matter. To switch it on for a regular page, add `math: true`.
- Use letters, digits, `:` and `-` in labels and citation keys.

The machinery lives in `_plugins/blog_tools.rb` (math, citations, figure numbering),
`_includes/figure.html` and `_includes/math.html` (MathJax).

## Previewing locally (optional)

With Ruby 3.1 or newer installed (on a Mac: `brew install ruby`):

```sh
bundle install
bundle exec jekyll serve --drafts
```

Then open <http://localhost:4000>. `--drafts` also shows the files in `_drafts/`. Jekyll rebuilds whenever
you save a file; refresh the page to see the change.
