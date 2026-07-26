# Source and interpretation boundary

- Author: [cephalopod (@macrocephalopod)](https://x.com/macrocephalopod)
- Thread head:
  [1806436278067470524](https://x.com/macrocephalopod/status/1806436278067470524)
- Published: 2024-06-27 21:15:45 UTC
- Length: 9 tweets, 5 attached handwritten images
- Retrieved: 2026-07-26 via the public FxTwitter API and a Thread Reader
  unroll; images pulled from `pbs.twimg.com` at `?name=orig`

This file is the frozen record of what the source actually says. Everything
this project adds: the simulation, the counterexamples, the trade-frequency
check, the estimator-noise analysis: is labelled as such and lives in
[`README.md`](README.md), with the raw output under
[`results/`](results/).

---

## 1. Thread text, verbatim

> **1/9**: Correlation between your signal and future returns is an
> important metric in quant trading. But what is a "good" correlation?
> Here's a simple way to think about it.

> **2/9**: We'll use a simple model where future returns y over some time
> period tau are normally distributed with a mean of beta \* x and a daily
> volatility of sigma (here x is a signal with std deviation 1)
> *[image 1]*

> **3/9**: We can easily work out the correlation between signal and returns
> and use that to express beta as a function of correlation, volatility and
> forecast horizon.
> *[image 2]*

> **4/9**: The key insight is that it should be easy to find signals that
> are not profitable if you take trading costs into account, since you won't
> be able to action them anyway.
>
> If we require that even a three standard deviation signal is unprofitable
> then we can bound the correlation —
> *[image 3]*

> **5/9**: What does that tell us? Say that we are interested in a stock
> with 3% daily volatility, trading cost of 5 bps and a forecast horizon of
> one day. Then we expect to easily find signals with a correlation of 0.5%
> with future returns
> *[image 4]*

> **6/9**: If we use factor hedging to remove non-idiosyncratic risk, we
> might halve the volatility and double the cost to trade (since we need to
> trade the factor hedge as well) so we expect to be able to easily find
> signals with 2% correlation with future idiosyncratic return.
> *[no image]*

> **7/9**: Alternatively if we are trying to predict a short term (1 minute)
> fx return which costs 0.2bps to trade and has a daily vol of 0.3% then we
> expect to easily find alphas with a much higher correlation of 8.5%
> *[image 5]*

> **8/9**: You can think of these as absolute minimum correlations, you need
> to exceed these to be able to trade profitably. As a rule of thumb, a
> correlation which is 1.5x the minimum would be ok (you will have a trade to
> do in about 5% of periods) and 2x the minimum is very good (you will be
> able to trade 20-30% of the time)

> **9/9**: This is one of the many ways that you can extend the law of
> active management to be more relevant to real world trading, a topic for
> another time maybe.

Engagement at retrieval: 1,049 likes, 1,911 bookmarks, 212,278 views.

## 2. Images, transcribed

Five photographs of handwritten notes. Transcription is verbatim; nothing is
tidied or completed. The same applies to the tweet text above, including its
punctuation: the trailing dash at the end of tweet 4 is the author's.

**Image 1** (tweet 2/9): the model:

$$y = \beta\cdot x + \sigma\sqrt{\tau}\cdot\varepsilon$$

$$\operatorname{Var}(x)=1,\qquad
  \operatorname{Var}(\varepsilon)=1,\qquad
  \operatorname{Cov}(x,\varepsilon)=0$$

**Image 2** (tweet 3/9): correlation, then invert for $\beta$:

$$\operatorname{Corr}(x,y)
  =\frac{\operatorname{Cov}(x,y)}
        {\bigl(\operatorname{Var}(x)\operatorname{Var}(y)\bigr)^{1/2}}
  =\frac{\beta}{(\sigma^{2}\tau)^{1/2}}
  =\frac{\beta}{\sigma\sqrt{\tau}}$$

$$\Rightarrow\quad \beta=\rho\cdot\sigma\cdot\sqrt{\tau}$$

**Image 3** (tweet 4/9): the cost bound. The annotation "cost to trade"
points at $c$:

$$3\beta\le c
  \quad\Rightarrow\quad
  3\rho\,\sigma\sqrt{\tau}\le c
  \quad\Rightarrow\quad
  \rho\le\frac{c}{3\sigma\sqrt{\tau}}$$

**Image 4** (tweet 5/9): the equity example:

$$\rho\le\frac{5\times10^{-4}}{3\times3\times10^{-2}\times\sqrt{1}}=0.5\%$$

**Image 5** (tweet 7/9): the FX example:

$$\rho\le\frac{0.00002}{3\times0.003\times\sqrt{\tfrac{1}{1440}}}=8.5\%$$

## 3. What the source states

1. A single-signal linear model for horizon-$\tau$ return with a
   unit-variance signal, Gaussian residual, signal and residual independent.
2. $\sigma$ is a **daily** volatility and $\tau$ is measured **in days**, so
   the residual scales as $\sigma\sqrt{\tau}$ (image 5 uses
   $\tau=1/1440$ for one minute, confirming the day unit).
3. $\rho=\beta/(\sigma\sqrt\tau)$, hence $\beta=\rho\sigma\sqrt\tau$.
4. A threshold: a signal is unactionable if even a $3\sigma$ realisation of
   $x$ fails to cover the cost to trade, $3\beta\le c$.
5. That threshold rearranged into a correlation floor
   $\rho_{\min}=c/(3\sigma\sqrt\tau)$, described as the level of correlation
   described as the level that should be **easy to find**, and therefore the
   level that must be **exceeded** to trade at all.
6. Three worked numbers: equity $0.5\%$, factor-hedged equity $2\%$
   (stated arithmetically in prose, no image), one-minute FX $8.5\%$.
7. Two rules of thumb: $1.5\times\rho_{\min}$ is "ok" and corresponds to a
   trade in about $5\%$ of periods; $2\times\rho_{\min}$ is "very good" and
   corresponds to trading $20$ to $30\%$ of the time.
8. A closing claim of lineage: this is an extension of the law of active
   management.

## 4. What the source does not state

- Whether $\rho$ is a Pearson correlation on raw returns, on residual
  returns, in-sample or out-of-sample, per-asset or pooled.
- The distribution of $x$ beyond $\operatorname{Var}(x)=1$. The $3\sigma$
  criterion and both rules of thumb are only interpretable if $x$ is
  Gaussian, but Gaussianity of $x$ is never asserted: only Gaussianity of
  the *return given the signal*.
- Whether $c$ is a one-way cost, a round trip, a half-spread, or an
  impact-inclusive number. "Cost to trade" with "double the cost to trade"
  for a hedge leg suggests a fixed per-round-trip proportional cost, but this
  is not pinned down.
- The trading rule that turns $\rho$ into P&L. The floor is derived from a
  single-period profitability comparison; no position-sizing, no holding
  period, no inventory or no-trade-band model is given.
- How the stated trade frequencies ($5\%$, $20$ to $30\%$) are computed.
- Which "law of active management" statement is being extended, and how.
- Any data, code, or empirical calibration. The thread is entirely
  analytical.

## 5. Verification verdict

Derivations, counterexamples and full detail are in
[`README.md`](README.md); the machine-readable output each verdict rests on is
in [`results/`](results/).

| # | Claim | Verdict |
|---|---|---|
| 1 | $\operatorname{Corr}(x,y)=\beta/(\sigma\sqrt\tau)$ | **Approximate.** Exactly $\beta/\sqrt{\beta^{2}+\sigma^{2}\tau}$. The error is $O(\rho^{2})$: at the largest $\rho$ in the thread ($8.5\%$) it is $0.36\%$ relative. Immaterial. |
| 2 | $\beta=\rho\sigma\sqrt\tau$ | **Correct** under claim 1's approximation; exactly $\beta=\rho\sigma\sqrt{\tau}/\sqrt{1-\rho^{2}}$. |
| 3 | $3\beta\le c\Rightarrow\rho\le c/(3\sigma\sqrt\tau)$ | **Correct** algebra. The choice of $3$ is a convention, and it silently assumes $x$ is Gaussian for "three standard deviations" to mean "essentially never". |
| 4 | Equity: $0.5\%$ | **Correct to one figure.** Exact value $0.556\%$; the thread rounds down by $11\%$ relative. |
| 5 | Factor-hedged: $2\%$ | **Correct to one figure.** Exact value $2.222\%$, again $11\%$ low. Verified symbolically to be exactly $4\times$ the unhedged floor (cost $\times2$, vol $\times\tfrac12$). |
| 6 | 1-min FX: $8.5\%$ | **Correct.** Exact value $8.433\%$, $0.8\%$ high. |
| 7 | $1.5\times\rho_{\min}\Rightarrow$ trade $\approx5\%$ of periods | **Correct.** Model-implied $P(\lvert x\rvert>2)=4.55\%$. |
| 8 | $2\times\rho_{\min}\Rightarrow$ trade $20$ to $30\%$ of periods | **Does not reproduce.** Model-implied $P(\lvert x\rvert>1.5)=13.36\%$. Reaching $20\%$ needs $2.34\times$ and $30\%$ needs $2.89\times\rho_{\min}$. |
| 8b | "you need to exceed these to be able to trade profitably" | **Too strong.** Under the thread's own model net P&L is $\beta\,\mathbb{E}[(\lvert x\rvert-k)^{+}]>0$ for every $\beta>0$: a Gaussian signal clears any finite band eventually. The floor is a relevance threshold, not a profitability one. |
| 9 | Lineage: extends the law of active management | **Supported, to leading order.** $\rho$ is the information coefficient. The exact per-period Sharpe of the proportional rule is $\rho/\sqrt{1+\rho^{2}}$, which equals $\rho$ to $O(\rho^{3})$, so annualised $IR=\rho\sqrt{BR}$ recovers Grinold's law as an approximation valid at small $\rho$, with the cost floor as the addition. |

Claim 8 is the only arithmetic that fails, and it fails in the optimistic
direction. Claim 8b is a matter of framing rather than arithmetic, and the
correction strengthens the result: what dies below the floor is not profit but
relevance, which survives the objection that one could simply wait for a larger
signal.

Everything structural in the thread survives. What it omits: the distributional
assumption on $x$, the encoding-dependence of a Pearson correlation, the fact
that $\sigma$ and $c$ are averages over states the alpha does not weight
equally, the signal's decay rate, and the fact that a correlation just above the
floor still captures only a few percent of the gross signal value: is where
this project spends its effort.

## 6. Retrieval provenance

```
GET https://api.fxtwitter.com/macrocephalopod/status/<id>   # 9 ids, text + media
GET https://threadreaderapp.com/thread/1806436278067470524.html   # ordering
GET https://pbs.twimg.com/media/GRHAOjIWIAA1jN7.jpg?name=orig     # image 1
GET https://pbs.twimg.com/media/GRHAPB7WkAAwxmb.jpg?name=orig     # image 2
GET https://pbs.twimg.com/media/GRHAPeMWgAAYU1q.jpg?name=orig     # image 3
GET https://pbs.twimg.com/media/GRHAP_HXMAAU3_N.jpg?name=orig     # image 4
GET https://pbs.twimg.com/media/GRHAQn8W0AA6uCo.jpg?name=orig     # image 5
```

Tweet ids in thread order: `1806436278067470524`, `1806436286527397893`,
`1806436294127472789`, `1806436302700605518`, `1806436311009530200`,
`1806436314536870067`, `1806436321138733550`, `1806436324796403948`,
`1806436327585427806`.
