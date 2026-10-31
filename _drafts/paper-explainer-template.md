---
title: "Temperature scaling, explained"
description: "Why accurate neural networks are often overconfident, how to measure it, and how a single parameter fixed after training repairs most of it."
macros:
  acc: '\operatorname{acc}'
  conf: '\operatorname{conf}'
  softmax: '\operatorname{softmax}'
---

{% comment %}
TEMPLATE FOR PAPER WRITE-UPS. This comment block is never shown on the site.
  ·
Publish:   copy this file to _posts/YYYY-MM-DD-short-name.md, edit, commit, push.
           Files in _drafts/ are never published. README.md has the full guide.
Summary:   `description` (front matter) is shown on the blog index and in search results.
Images:    put them in assets/blog/short-name/ (short-name = file name without date and .md).
  ·
Math       inline:  $x$  or  \(x\)            literal dollar sign:  \$
           display: $$ ... $$ on its own lines, \[ ... \], or an environment:
                    \begin{equation} ... \label{eq:name} \end{equation}   (also align, gather, multline)
           refer:   \eqref{eq:name} -> (1)      \ref{eq:name} -> 1
           macros:  `macros:` in the front matter (this post), `math_macros:` in _config.yml (all posts)
  ·
Figures    {% include figure.html src="plot.png" label="fig:name" caption="Markdown, $math$ and \cite{key} work here." %}
           optional: width="70%"  alt="..."   refer: Figure~\ref{fig:name}
  ·
Citations  add BibTeX to _bibliography/references.bib, then
           \cite{key} -> [1]     \cite{a,b} -> [1, 2]     \cite[Sec.~3]{key} -> [1, Sec. 3]     \citet{key} -> Guo et al. [1]
           The References section at the end is generated automatically.
  ·
Footnotes  text[^note]   and on its own line:   [^note]: The footnote.
Contents   the two-line "* Table of contents / {:toc}" block below builds a linked table of contents.
Labels     use letters, digits, ":" and "-" in labels and keys (e.g. fig:loss-curve, eq:elbo).
{% endcomment %}

Modern deep networks are accurate but often *overconfident*: when they report 90% confidence they are right far less often than 90% of the time \cite{guo2017calibration}. \citet{guo2017calibration} showed that a single scalar, a temperature $T$ fitted after training, removes most of this miscalibration without changing a single prediction. The question of whether a model's stated confidence can be trusted matters even more for large language models built on the Transformer \cite{vaswani2017attention, dey2026sure}.

* Table of contents
{:toc}

## Measuring calibration

A classifier maps an input to logits $z \in \R^K$, predicts $\hat{y} = \argmax_k z_k$, and reports the confidence $\hat{p} = \max_k \softmax(z)_k$. It is *calibrated* if, among all predictions made with confidence $p$, a fraction $p$ turn out to be correct:

$$
\mathbb{P}\left(\hat{y} = y \mid \hat{p} = p\right) = p \qquad \text{for all } p \in [0, 1].
$$

To measure the gap on a finite test set, group the $n$ predictions into $M$ equal-width confidence bins $B_1, \dots, B_M$ and compare accuracy with average confidence in each bin. The weighted average of these gaps is the *expected calibration error* (ECE):

\begin{equation}
\mathrm{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \, \bigl| \acc(B_m) - \conf(B_m) \bigr| \label{eq:ece}
\end{equation}

Figure~\ref{fig:reliability} shows the two terms of \eqref{eq:ece} bin by bin as a *reliability diagram*. Bars below the diagonal mean the model is more confident than it is accurate.

{% include figure.html src="reliability.png" label="fig:reliability" caption="Reliability diagrams for a simulated, overconfident 10-class classifier, before (left) and after (right) temperature scaling. Bars show accuracy per confidence bin, and the hatched area is the gap to the bin's average confidence. Synthetic data, for illustration." %}

## Temperature scaling

The fix divides every logit by one temperature $T > 0$ before the softmax. Dividing by $T$ does not change which logit is largest, so accuracy is untouched; only the confidence moves:

\begin{align}
\hat{q}_i &= \max_k \, \softmax\!\left(z_i / T\right)_k, \label{eq:temp} \\
T^* &= \argmin_{T > 0} \; -\sum_{(z_i,\, y_i) \in \mathcal{V}} \log \softmax\!\left(z_i / T\right)_{y_i}. \label{eq:nll}
\end{align}

A temperature $T > 1$ softens overconfident predictions in \eqref{eq:temp}, and \eqref{eq:nll} chooses $T$ by minimizing the negative log-likelihood on a held-out validation set $\mathcal{V}$. In our simulation the likelihood-optimal $T^* = 1.62$ lands almost exactly where ECE is smallest (Figure~\ref{fig:temperature}), and ECE drops from 0.153 to 0.010.

{% include figure.html src="ece-vs-temperature.png" label="fig:temperature" caption="ECE as a function of the temperature $T$ for the same simulated model. The marked point is $T^*$ from \eqref{eq:nll}, fitted on likelihood alone." width="85%" %}

Fitting $T$ takes a few lines of code:

```python
import torch
import torch.nn.functional as F

def fit_temperature(logits, labels):
    """logits: [n, K] validation logits; labels: [n] class indices."""
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t) stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()
```

## Takeaways

- Accuracy and calibration are different properties: in \citet{guo2017calibration}, deeper and wider networks were more accurate but less well calibrated.
- One parameter, fitted after training, removes most of the miscalibration here, and it never changes the predicted class.
- The fitted temperature is only as good as the validation set: under distribution shift, the right $T$ can change.[^shift]

[^shift]: Footnotes like this one are good for asides that would interrupt the main argument.
