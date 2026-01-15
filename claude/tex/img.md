# Programmatic Figure Generation Guide

This document contains scripts to generate all figures for the thesis.

## Approach

- **Figures 2.1, 2.2, 2.3, 2.4, 2.5, 2.6**: Use TikZ (LaTeX) for architecture diagrams and visualizations
- **Figures 3.2, 3.3**: Use Python (matplotlib) for comparison plots and dataset examples

---

## Figure 2.1: FNO Architecture

### TikZ Code (for LaTeX document)

```latex
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.5cm,
    layer/.style={rectangle, draw, minimum width=3cm, minimum height=1cm, align=center},
    arrow/.style={->, >=stealth, thick}
]

% Input
\node[layer, fill=blue!20] (input) {Input\\$u(x)$\\dim: $s \times s$};

% Lifting layer
\node[layer, fill=green!20, below of=input] (lift) {Lifting Layer (P)\\$v_0 = Pu$\\dim: $s \times s \times d$};

% Fourier layers
\node[layer, fill=orange!20, below of=lift] (f1) {Fourier Layer 1\\Spectral Conv + Act};
\node[layer, fill=orange!20, below of=f1] (f2) {Fourier Layer 2\\Spectral Conv + Act};
\node[layer, fill=orange!20, below of=f2] (f3) {Fourier Layer 3\\Spectral Conv + Act};
\node[layer, fill=orange!20, below of=f3] (f4) {Fourier Layer 4\\Spectral Conv + Act};

% Projection layer
\node[layer, fill=green!20, below of=f4] (proj) {Projection Layer (Q)\\$y = Qv_4$\\dim: $s \times s \times 1$};

% Output
\node[layer, fill=blue!20, below of=proj] (output) {Output\\$\hat{u}(x)$\\dim: $s \times s$};

% Arrows
\draw[arrow] (input) -- (lift);
\draw[arrow] (lift) -- (f1);
\draw[arrow] (f1) -- (f2);
\draw[arrow] (f2) -- (f3);
\draw[arrow] (f3) -- (f4);
\draw[arrow] (f4) -- (proj);
\draw[arrow] (proj) -- (output);

% Skip connections
\draw[arrow, dashed, red] (f1.east) to[out=0, in=0] node[right, xshift=0.2cm] {skip} (f2.east);
\draw[arrow, dashed, red] (f2.east) to[out=0, in=0] (f3.east);
\draw[arrow, dashed, red] (f3.east) to[out=0, in=0] (f4.east);

\end{tikzpicture}
\caption{Fourier Neural Operator architecture with lifting layer (P), four Fourier layers with spectral convolution and skip connections, and projection layer (Q).}
\label{fig:fno_architecture}
\end{figure}
```

**Required LaTeX packages:**
```latex
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric}
```

---

## Figure 2.2: Spectral Convolution

### TikZ Code

```latex
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=0.8cm,
    box/.style={rectangle, draw, minimum width=2.2cm, minimum height=2.2cm, align=center, font=\small},
    smallbox/.style={rectangle, draw, minimum width=1.2cm, minimum height=1.2cm, align=center, font=\scriptsize},
    arrow/.style={->, >=stealth, thick},
    label/.style={font=\footnotesize, align=center}
]

% ===== Input u(x) in spatial domain =====
\node[box, fill=blue!15] (input) {
    \textbf{Input}\\[2pt]
    $u(x)$\\[4pt]
    \begin{tikzpicture}[scale=0.4]
        \foreach \x in {0,0.3,...,1.5} {
            \foreach \y in {0,0.3,...,1.5} {
                \pgfmathsetmacro{\c}{50+40*sin(\x*180)*cos(\y*180)}
                \fill[blue!\c!white] (\x,\y) rectangle (\x+0.3,\y+0.3);
            }
        }
        \draw[black] (0,0) rectangle (1.8,1.8);
    \end{tikzpicture}
};
\node[label, below=0.1cm of input] {Spatial Domain\\$s \times s$};

% ===== Fourier Space (full) =====
\node[box, fill=purple!15, right=2.5cm of input] (fourier) {
    \textbf{Fourier}\\[2pt]
    $\hat{u}(k)$\\[4pt]
    \begin{tikzpicture}[scale=0.4]
        % Full Fourier space (darker outside)
        \fill[purple!40] (0,0) rectangle (1.8,1.8);
        % Kept modes in center (brighter)
        \fill[yellow!60] (0.5,0.5) rectangle (1.3,1.3);
        \draw[green!70!black, line width=1pt] (0.5,0.5) rectangle (1.3,1.3);
        \draw[black] (0,0) rectangle (1.8,1.8);
    \end{tikzpicture}
};
\node[label, below=0.1cm of fourier] {Spectral Domain\\$k_{\max} \times k_{\max}$ modes};

% ===== Mode Truncation annotation =====
\node[above=0.3cm of fourier, font=\scriptsize, text=green!50!black] {Keep $12 \times 12$ modes};

% ===== Learnable Weights R =====
\node[box, fill=orange!20, right=2.5cm of fourier] (weights) {
    \textbf{Filtered}\\[2pt]
    $R \odot \hat{u}(k)$\\[4pt]
    \begin{tikzpicture}[scale=0.4]
        % Only kept modes shown
        \fill[gray!20] (0,0) rectangle (1.8,1.8);
        \fill[orange!50] (0.5,0.5) rectangle (1.3,1.3);
        \draw[red!70!black, line width=1pt] (0.5,0.5) rectangle (1.3,1.3);
        \draw[black] (0,0) rectangle (1.8,1.8);
    \end{tikzpicture}
};
\node[label, below=0.1cm of weights] {Learnable $R$\\(complex weights)};

% ===== Output v(x) =====
\node[box, fill=green!15, right=2.5cm of weights] (output) {
    \textbf{Output}\\[2pt]
    $v(x)$\\[4pt]
    \begin{tikzpicture}[scale=0.4]
        \foreach \x in {0,0.3,...,1.5} {
            \foreach \y in {0,0.3,...,1.5} {
                \pgfmathsetmacro{\c}{30+50*sin(\x*120)*cos(\y*150)}
                \fill[green!\c!white] (\x,\y) rectangle (\x+0.3,\y+0.3);
            }
        }
        \draw[black] (0,0) rectangle (1.8,1.8);
    \end{tikzpicture}
};
\node[label, below=0.1cm of output] {Spatial Domain\\$s \times s$};

% ===== Arrows connecting boxes =====
\draw[arrow, line width=1.5pt, blue!70] (input.east) -- node[above, font=\small\bfseries] {FFT} node[below, font=\footnotesize] {$\mathcal{F}$} (fourier.west);
\draw[arrow, line width=1.5pt, purple!70] (fourier.east) -- node[above, font=\small\bfseries] {$\odot$} node[below, font=\footnotesize] {Multiply} (weights.west);
\draw[arrow, line width=1.5pt, green!60!black] (weights.east) -- node[above, font=\small\bfseries] {IFFT} node[below, font=\footnotesize] {$\mathcal{F}^{-1}$} (output.west);

% ===== Bottom equation =====
\node[below=1.5cm of weights, font=\normalsize] {
    $\mathcal{K}(u)(x) = \mathcal{F}^{-1}\Big( R \cdot \mathcal{F}(u) \Big)(x)$
};

% ===== Legend for mode truncation =====
\node[right=0.3cm of output, text width=2.5cm, font=\scriptsize, align=left] {
    \textcolor{green!50!black}{\rule{0.3cm}{0.3cm}} Kept modes\\[2pt]
    \textcolor{gray!50}{\rule{0.3cm}{0.3cm}} Truncated
};

\end{tikzpicture}
\caption{Spectral convolution in the Fourier Neural Operator. The input function $u(x)$ is transformed to Fourier space via FFT, where only $k_{\max} \times k_{\max}$ low-frequency modes are retained. These modes are multiplied element-wise with learnable complex weights $R$, then transformed back to spatial domain via IFFT.}
\label{fig:spectral_convolution}
\end{figure}
```

**Required LaTeX packages:**
```latex
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, calc}
```

---

## Figure 2.3: MC Dropout Inference

### TikZ Code

```latex
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1cm and 1.5cm,
    model/.style={rectangle, draw, minimum width=2cm, minimum height=1.5cm, align=center, fill=blue!20},
    pass/.style={rectangle, draw, minimum width=1.5cm, minimum height=1cm, align=center, fill=orange!20},
    result/.style={rectangle, draw, minimum width=2cm, minimum height=1cm, align=center, fill=green!20}
]

% Trained model
\node[model] (model) {Trained\\FNO Model\\(with Dropout)};

% Input
\node[above=0.5cm of model] (input) {$u(x)$};
\draw[->, thick] (input) -- (model);

% Multiple forward passes
\node[pass, below right=0.5cm and 0.5cm of model] (p1) {Pass 1\\$p=0.1$};
\node[pass, right=0.3cm of p1] (p2) {Pass 2\\$p=0.1$};
\node[pass, right=0.3cm of p2] (p3) {...};
\node[pass, right=0.3cm of p3] (p30) {Pass 30\\$p=0.1$};

% Arrows from model to passes
\draw[->, thick] (model.east) -| (p1.north);
\draw[->, thick] (model.east) -| (p2.north);
\draw[->, thick] (model.east) -| (p3.north);
\draw[->, thick] (model.east) -| (p30.north);

% Outputs
\node[below=0.5cm of p1, font=\small] (o1) {$\hat{y}_1$};
\node[below=0.5cm of p2, font=\small] (o2) {$\hat{y}_2$};
\node[below=0.5cm of p3, font=\small] (o3) {...};
\node[below=0.5cm of p30, font=\small] (o30) {$\hat{y}_{30}$};

\draw[->, thick] (p1) -- (o1);
\draw[->, thick] (p2) -- (o2);
\draw[->, thick] (p30) -- (o30);

% Aggregation
\node[result, below=1cm of p2] (mean) {Mean:\\$\mu = \frac{1}{T}\sum_{t=1}^{T} \hat{y}_t$};
\node[result, below=0.3cm of mean] (var) {Variance:\\$\sigma^2 = \frac{1}{T}\sum_{t=1}^{T} (\hat{y}_t - \mu)^2$};

\draw[->, thick, dashed] (o1.south) -- ++(0,-0.3) -| (mean.north);
\draw[->, thick, dashed] (o2.south) -- ++(0,-0.3) -| (mean.north);
\draw[->, thick, dashed] (o30.south) -- ++(0,-0.3) -| (mean.north);

% Dropout mask visualization
\node[below=0.1cm of p1, font=\tiny, text=red] {mask $m_1$};
\node[below=0.1cm of p2, font=\tiny, text=red] {mask $m_2$};
\node[below=0.1cm of p30, font=\tiny, text=red] {mask $m_{30}$};

% Annotation
\node[right=0.5cm of var, text width=3cm, font=\small, align=left] {
    \textbf{Epistemic} uncertainty\\
    (model uncertainty)
};

\end{tikzpicture}
\caption{Monte Carlo Dropout inference with T=30 stochastic forward passes. Different dropout masks lead to diverse predictions, quantifying epistemic uncertainty.}
\label{fig:mc_dropout}
\end{figure}
```

---

## Figure 2.4: Deep Ensembles

### TikZ Code

```latex
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.5cm and 1cm,
    model/.style={rectangle, draw, minimum width=2cm, minimum height=2.5cm, align=center, rounded corners},
    result/.style={rectangle, draw, minimum width=2.5cm, minimum height=1cm, align=center, fill=green!20}
]

% Input (shared)
\node (input) {$u(x)$};

% Five models with different colors
\node[model, fill=blue!20, below right=0.5cm and -1cm of input] (m1) {
    \textbf{Model 1}\\
    Init: $\theta_1^{(0)}$\\
    Data: $\mathcal{D}_1$\\
    $\downarrow$\\
    $\hat{y}_1$
};

\node[model, fill=red!20, right=0.5cm of m1] (m2) {
    \textbf{Model 2}\\
    Init: $\theta_2^{(0)}$\\
    Data: $\mathcal{D}_2$\\
    $\downarrow$\\
    $\hat{y}_2$
};

\node[model, fill=yellow!20, right=0.5cm of m2] (m3) {
    \textbf{Model 3}\\
    Init: $\theta_3^{(0)}$\\
    Data: $\mathcal{D}_3$\\
    $\downarrow$\\
    $\hat{y}_3$
};

\node[model, fill=purple!20, right=0.5cm of m3] (m4) {
    \textbf{Model 4}\\
    Init: $\theta_4^{(0)}$\\
    Data: $\mathcal{D}_4$\\
    $\downarrow$\\
    $\hat{y}_4$
};

\node[model, fill=cyan!20, right=0.5cm of m4] (m5) {
    \textbf{Model 5}\\
    Init: $\theta_5^{(0)}$\\
    Data: $\mathcal{D}_5$\\
    $\downarrow$\\
    $\hat{y}_5$
};

% Input arrows
\draw[->, thick] (input) -| (m1.north);
\draw[->, thick] (input) -| (m2.north);
\draw[->, thick] (input) -| (m3.north);
\draw[->, thick] (input) -| (m4.north);
\draw[->, thick] (input) -| (m5.north);

% Aggregation
\node[result, below=1.5cm of m3] (mean) {
    Mean: $\mu = \frac{1}{M}\sum_{m=1}^{M} \hat{y}_m$
};

\node[result, below=0.3cm of mean] (var) {
    Variance: $\sigma^2 = \frac{1}{M}\sum_{m=1}^{M} (\hat{y}_m - \mu)^2$
};

% Aggregation arrows
\draw[->, thick, dashed] (m1.south) -- ++(0,-0.3) -| (mean.north);
\draw[->, thick, dashed] (m2.south) -- ++(0,-0.3) -| (mean.north);
\draw[->, thick, dashed] (m3.south) -- ++(0,-0.3) -| (mean.north);
\draw[->, thick, dashed] (m4.south) -- ++(0,-0.3) -| (mean.north);
\draw[->, thick, dashed] (m5.south) -- ++(0,-0.3) -| (mean.north);

% Annotation
\node[above=0.2cm of m3, text width=4cm, font=\small, align=center] {
    \textbf{Independent Training}\\
    Different initializations \& data splits
};

\node[right=0.5cm of var, text width=3cm, font=\small, align=left] {
    \textbf{Diversity} $\rightarrow$\\
    Better uncertainty\\
    quantification
};

\end{tikzpicture}
\caption{Deep Ensemble with M=5 independently trained models. Different random initializations and data splits lead to diverse predictions that capture epistemic uncertainty.}
\label{fig:deep_ensemble}
\end{figure}
```

---

## Figure 2.5: Standard Regression vs Evidential Regression

### TikZ Code

```latex
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=0.8cm and 1.2cm,
    box/.style={rectangle, draw, minimum width=2cm, minimum height=1.2cm, align=center, rounded corners=3pt},
    smallbox/.style={rectangle, draw, minimum width=1.5cm, minimum height=0.8cm, align=center, rounded corners=2pt, font=\small},
    arrow/.style={->, >=stealth, thick},
    title/.style={font=\bfseries\large}
]

% ==================== LEFT SIDE: Standard Regression ====================
\node[title] at (2.5, 5) {(a) Standard Regression};

% Input
\node[box, fill=blue!20] (std_input) at (0, 3.5) {
    \textbf{Input}\\
    $x$
};

% Neural Network
\node[box, fill=orange!30, minimum height=2cm, minimum width=2.5cm] (std_net) at (2.5, 3.5) {
    \textbf{Neural}\\
    \textbf{Network}\\
    $f_\theta(x)$
};

% Single Output
\node[box, fill=green!30] (std_output) at (5, 3.5) {
    \textbf{Output}\\
    $\hat{y}$
};

% Arrows
\draw[arrow, line width=1.5pt] (std_input) -- (std_net);
\draw[arrow, line width=1.5pt] (std_net) -- (std_output);

% Loss function box
\node[draw, fill=red!15, rounded corners, text width=4cm, align=center] (std_loss) at (2.5, 1.5) {
    \textbf{MSE Loss}\\[3pt]
    $\mathcal{L} = (y - \hat{y})^2$
};

\draw[arrow, dashed, gray] (std_output.south) -- ++(0,-0.5) -| (std_loss.north);

% Output visualization - point estimate
\begin{scope}[shift={(0, -1)}]
    \draw[->] (0,0) -- (5,0) node[right, font=\small] {$y$};
    \draw[->] (0,0) -- (0,1.5) node[above, font=\small] {$p(y)$};

    % Single point (Dirac delta approximation)
    \fill[green!60] (2.5,0) circle (4pt);
    \draw[green!70!black, line width=2pt] (2.5,0) -- (2.5,1.2);
    \node[font=\footnotesize, above] at (2.5,1.2) {$\hat{y}$};

    % Label
    \node[font=\footnotesize, text=gray] at (2.5, -0.5) {Point estimate only};
\end{scope}

% Characteristics box
\node[draw, fill=gray!10, rounded corners, text width=4.5cm, align=left, font=\footnotesize] at (2.5, -3) {
    \textbf{Characteristics:}\\
    $\bullet$ Single point prediction\\
    $\bullet$ No uncertainty estimate\\
    $\bullet$ Assumes homoscedastic noise\\
    $\bullet$ Cannot detect OOD data
};

% ==================== RIGHT SIDE: Evidential Regression ====================
\node[title] at (10.5, 5) {(b) Evidential Regression};

% Input
\node[box, fill=blue!20] (evd_input) at (8, 3.5) {
    \textbf{Input}\\
    $x$
};

% Neural Network
\node[box, fill=orange!30, minimum height=2cm, minimum width=2.5cm] (evd_net) at (10.5, 3.5) {
    \textbf{Neural}\\
    \textbf{Network}\\
    $f_\theta(x)$
};

% Four output heads
\node[smallbox, fill=cyan!30] (gamma) at (13.5, 4.5) {$\gamma$};
\node[smallbox, fill=lime!30] (nu) at (13.5, 3.8) {$\nu$};
\node[smallbox, fill=yellow!40] (alpha) at (13.5, 3.1) {$\alpha$};
\node[smallbox, fill=pink!40] (beta) at (13.5, 2.4) {$\beta$};

% Arrows from network to heads
\draw[arrow, line width=1.5pt] (evd_input) -- (evd_net);
\draw[arrow] (evd_net.east) -- ++(0.3,0) |- (gamma.west);
\draw[arrow] (evd_net.east) -- ++(0.3,0) |- (nu.west);
\draw[arrow] (evd_net.east) -- ++(0.3,0) |- (alpha.west);
\draw[arrow] (evd_net.east) -- ++(0.3,0) |- (beta.west);

% NIG output
\node[box, fill=purple!25, minimum width=2.2cm] (nig_out) at (16, 3.5) {
    \textbf{NIG}\\
    $(\gamma,\nu,\alpha,\beta)$
};

\draw[arrow] (gamma.east) -- ++(0.2,0) |- (nig_out.west);
\draw[arrow] (nu.east) -- ++(0.2,0) |- (nig_out.west);
\draw[arrow] (alpha.east) -- ++(0.2,0) |- (nig_out.west);
\draw[arrow] (beta.east) -- ++(0.2,0) |- (nig_out.west);

% Loss function box
\node[draw, fill=red!15, rounded corners, text width=5cm, align=center] (evd_loss) at (10.5, 1.5) {
    \textbf{Evidential Loss}\\[3pt]
    $\mathcal{L} = \mathcal{L}_{\text{NLL}} + \lambda \cdot \mathcal{L}_{\text{reg}}$
};

\draw[arrow, dashed, gray] (nig_out.south) -- ++(0,-0.5) -| (evd_loss.north);

% Output visualization - distribution
\begin{scope}[shift={(8, -1)}]
    \draw[->] (0,0) -- (5,0) node[right, font=\small] {$y$};
    \draw[->] (0,0) -- (0,1.5) node[above, font=\small] {$p(y)$};

    % Student-t distribution (bell curve)
    \fill[purple!20, smooth] plot coordinates {
        (0.5,0) (0.5,0.05) (1,0.15) (1.5,0.4) (2,0.85) (2.3,1.1) (2.5,1.2)
        (2.7,1.1) (3,0.85) (3.5,0.4) (4,0.15) (4.5,0.05) (4.5,0)
    } -- cycle;
    \draw[purple!70, line width=1.5pt, smooth] plot coordinates {
        (0.5,0.05) (1,0.15) (1.5,0.4) (2,0.85) (2.3,1.1) (2.5,1.2)
        (2.7,1.1) (3,0.85) (3.5,0.4) (4,0.15) (4.5,0.05)
    };

    % Mean line
    \draw[red, dashed, thick] (2.5,0) -- (2.5,1.35);
    \node[font=\footnotesize, red, above] at (2.5,1.35) {$\gamma$};

    % Uncertainty band
    \draw[<->, green!60!black, thick] (1.5,0.5) -- (3.5,0.5);
    \node[font=\tiny, green!60!black, above] at (2.5,0.5) {$\sigma^2_{\text{ale}} + \sigma^2_{\text{epi}}$};

    % Label
    \node[font=\footnotesize, text=purple!70] at (2.5, -0.5) {Full predictive distribution};
\end{scope}

% Characteristics box
\node[draw, fill=gray!10, rounded corners, text width=5cm, align=left, font=\footnotesize] at (10.5, -3) {
    \textbf{Characteristics:}\\
    $\bullet$ Distribution over predictions\\
    $\bullet$ Aleatoric: $\sigma^2_{\text{ale}} = \frac{\beta}{\alpha-1}$\\
    $\bullet$ Epistemic: $\sigma^2_{\text{epi}} = \frac{\beta}{\nu(\alpha-1)}$\\
    $\bullet$ Single forward pass\\
    $\bullet$ OOD detection capability
};

% ==================== Comparison Arrow ====================
\draw[<->, line width=2pt, red!60] (5.8, 3.5) -- (7.2, 3.5);
\node[font=\small, text=red!60, above] at (6.5, 3.7) {vs};

% ==================== Bottom Summary ====================
\node[draw, fill=yellow!10, rounded corners, text width=14cm, align=center] at (8, -5) {
    \textbf{Key Difference:} Standard regression outputs a single value $\hat{y}$, while evidential regression outputs distribution parameters $(\gamma, \nu, \alpha, \beta)$ that encode both the prediction and its uncertainty in a single forward pass.
};

\end{tikzpicture}
\caption{Comparison of standard regression and evidential regression. (a) Standard regression outputs a single point estimate with MSE loss, providing no uncertainty information. (b) Evidential regression outputs four parameters of the Normal-Inverse-Gamma distribution, enabling decomposition of uncertainty into aleatoric (data noise) and epistemic (model uncertainty) components from a single forward pass.}
\label{fig:standard_vs_evidential}
\end{figure}
```

**Required LaTeX packages:**
```latex
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric, calc}
```

---

## Figure 2.6: Evidential FNO

### TikZ Code

```latex
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=0.8cm and 1cm,
    box/.style={rectangle, draw, minimum width=1.8cm, minimum height=1.2cm, align=center, rounded corners=3pt},
    smallbox/.style={rectangle, draw, minimum width=2.2cm, minimum height=0.9cm, align=center, rounded corners=2pt},
    fnobox/.style={rectangle, draw, minimum width=1.4cm, minimum height=1cm, align=center, rounded corners=2pt},
    arrow/.style={->, >=stealth, thick},
    label/.style={font=\footnotesize, align=center}
]

% ==================== PART 1: Network Architecture ====================
\node[font=\bfseries] at (7, 5.5) {Evidential FNO Architecture};

% Input
\node[box, fill=blue!20] (input) at (0, 3) {
    \textbf{Input}\\
    $u(x)$\\
    \footnotesize$(H \times W)$
};

% ==================== Detailed FNO Backbone ====================
% Outer box for FNO
\node[draw, fill=blue!5, minimum width=7cm, minimum height=4cm, rounded corners=5pt] (fno_outer) at (4.8, 3) {};
\node[font=\small\bfseries, anchor=north] at (4.8, 5.1) {FNO Backbone};

% Lifting layer
\node[fnobox, fill=green!25] (lift) at (2, 3) {
    \textbf{Lift}\\
    \footnotesize$P$
};

% Branch point after lifting
\coordinate (branch) at (3, 3);

% FFT (upper spectral path)
\node[fnobox, fill=blue!30] (fft) at (4, 4) {
    \textbf{FFT}\\
    \footnotesize$\mathcal{F}$
};

% Fourier weights (spectral conv)
\node[fnobox, fill=orange!40] (weights) at (5.5, 4) {
    \textbf{$R \odot$}\\
    \footnotesize modes
};

% IFFT
\node[fnobox, fill=purple!30] (ifft) at (7, 4) {
    \textbf{IFFT}\\
    \footnotesize$\mathcal{F}^{-1}$
};

% Local path (W) - lower spatial path
\node[fnobox, fill=yellow!30, minimum width=2.5cm] (local) at (5.5, 2) {
    \textbf{$W$} \footnotesize(spatial conv)
};

% Plus node for combining paths
\node[circle, draw, fill=white, minimum size=0.6cm, font=\bfseries] (plus) at (7.8, 2.5) {$+$};

% Layer indicator
\node[font=\tiny, text=gray] at (4.8, 1.2) {$\times 4$ Fourier Layers};

% Arrow from input to lift
\draw[arrow, line width=1.5pt] (input.east) -- (lift.west);

% Arrow from lift to branch point
\draw[line width=1.5pt] (lift.east) -- (branch);

% Branching arrows - BOTH paths start from same branch point
\draw[arrow, line width=1pt, rounded corners=3pt] (branch) |- (fft.west);
\draw[arrow, line width=1pt, rounded corners=3pt] (branch) |- (local.west);

% Spectral path arrows
\draw[arrow, line width=1pt] (fft) -- (weights);
\draw[arrow, line width=1pt] (weights) -- (ifft);

% Arrows to plus node
\draw[arrow, line width=1pt, rounded corners=3pt] (ifft.south) -- ++(0,-0.5) -| (plus.north);
\draw[arrow, line width=1pt, rounded corners=3pt] (local.east) -| (plus.south);

% ==================== Four Output Heads (Equal Size) ====================
\node[smallbox, fill=cyan!35] (gamma) at (10.5, 4.5) {$\gamma$ \footnotesize(mean)};
\node[smallbox, fill=lime!35] (nu) at (10.5, 3.5) {$\nu$ \footnotesize(evidence)};
\node[smallbox, fill=yellow!45] (alpha) at (10.5, 2.5) {$\alpha$ \footnotesize(shape)};
\node[smallbox, fill=pink!45] (beta) at (10.5, 1.5) {$\beta$ \footnotesize(scale)};

% Smooth curved arrows from plus node to heads
\draw[arrow, line width=1.2pt, rounded corners=8pt] (plus.east) -- (8.8, 2.5) -- (8.8, 4.5) -- (gamma.west);
\draw[arrow, line width=1.2pt, rounded corners=8pt] (plus.east) -- (8.8, 2.5) -- (8.8, 3.5) -- (nu.west);
\draw[arrow, line width=1.2pt, rounded corners=8pt] (plus.east) -- (8.8, 2.5) -- (alpha.west);
\draw[arrow, line width=1.2pt, rounded corners=8pt] (plus.east) -- (8.8, 2.5) -- (8.8, 1.5) -- (beta.west);

% ==================== NIG Distribution with Loss ====================
\node[box, fill=purple!25, minimum height=3.2cm, minimum width=3cm] (nig) at (14, 3) {
    \textbf{NIG Distribution}\\[3pt]
    $p(\mu, \sigma^2 | \boldsymbol{\theta})$\\[6pt]
    \footnotesize\textbf{Loss:}\\[2pt]
    \tiny$\mathcal{L} = \frac{1}{2}\log\frac{\pi}{\nu} - \alpha\log\Omega$\\[1pt]
    \tiny$+ (\alpha+\frac{1}{2})\log\big((y-\gamma)^2\nu + \Omega\big)$\\[1pt]
    \tiny$+ \log\frac{\Gamma(\alpha)}{\Gamma(\alpha+\frac{1}{2})}$
};

% Smooth arrows from heads to NIG
\draw[arrow, rounded corners=5pt] (gamma.east) -- ++(0.5,0) |- ($(nig.west)+(0,0.8)$);
\draw[arrow, rounded corners=5pt] (nu.east) -- ++(0.3,0) |- ($(nig.west)+(0,0.3)$);
\draw[arrow, rounded corners=5pt] (alpha.east) -- ++(0.3,0) |- ($(nig.west)+(0,-0.3)$);
\draw[arrow, rounded corners=5pt] (beta.east) -- ++(0.5,0) |- ($(nig.west)+(0,-0.8)$);

% Activation annotations
\node[font=\tiny, text=gray, align=center] at (10.5, 5.1) {linear};
\node[font=\tiny, text=gray, align=center] at (10.5, 0.9) {softplus};

% ==================== Key Equations Box (Compact) ====================
\node[draw, fill=yellow!10, rounded corners, text width=4.2cm, align=left, font=\footnotesize] at (3, -0.8) {
    \textbf{Key Equations:}\\[2pt]
    $\hat{y} = \gamma$\\
    $\sigma^2_{\text{ale}} = \frac{\beta}{\alpha - 1}$\\
    $\sigma^2_{\text{epi}} = \frac{\beta}{\nu(\alpha - 1)}$
};

% ==================== Student-t Predictive (Compact) ====================
\node[font=\small\bfseries] at (10, -0.2) {Student-t Predictive};

\begin{scope}[shift={(7.5,-1.8)}]
    % Axes
    \draw[->] (0,0) -- (4.5,0) node[right, font=\footnotesize] {$y$};
    \draw[->] (0,0) -- (0,2) node[above, font=\footnotesize] {$p(y)$};

    % Bell curve (Student-t approximation)
    \draw[blue!70, line width=1.5pt, smooth] plot coordinates {
        (0.2,0.03) (0.6,0.1) (1.1,0.35) (1.6,0.8) (2,1.4) (2.25,1.7)
        (2.5,1.4) (2.9,0.8) (3.4,0.35) (3.9,0.1) (4.3,0.03)
    };

    % Fill under curve
    \fill[blue!15, smooth] plot coordinates {
        (0.2,0) (0.2,0.03) (0.6,0.1) (1.1,0.35) (1.6,0.8) (2,1.4) (2.25,1.7)
        (2.5,1.4) (2.9,0.8) (3.4,0.35) (3.9,0.1) (4.3,0.03) (4.3,0)
    } -- cycle;

    % Mean line
    \draw[red, dashed, thick] (2.25,0) -- (2.25,1.85);
    \node[font=\footnotesize, red] at (2.25,1.95) {$\gamma$};

    % Uncertainty width
    \draw[<->, green!60!black, thick] (1.3,1) -- (3.2,1);
    \node[font=\tiny, green!60!black, above] at (2.25,1) {$\sigma^2$};
\end{scope}

\end{tikzpicture}
\caption{Evidential FNO framework. The FNO backbone processes input through a lifting layer, then branches into parallel paths: a spectral path (FFT $\rightarrow$ learnable weights $R$ $\rightarrow$ IFFT) and a spatial convolution path ($W$). Outputs combine at the $+$ node and feed into four projection heads producing NIG parameters $(\gamma, \nu, \alpha, \beta)$. The NIG loss enables single-pass uncertainty estimation with decomposition into aleatoric ($\sigma^2_{\text{ale}}$) and epistemic ($\sigma^2_{\text{epi}}$) components.}
\label{fig:evidential_framework}
\end{figure}
```

**Required LaTeX packages:**
```latex
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric, calc}
```

---

## Usage Instructions

### For Chapter 2 TikZ figures (2.1, 2.2, 2.3, 2.4, 2.5, 2.6):

1. Copy the TikZ code into your LaTeX document
2. Ensure you have the required packages in your preamble:
```latex
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric, calc}
```
3. Compile with pdflatex or xelatex

### For Chapter 3 Python figures (3.2, 3.3):

1. Save the Python scripts to files:
   - `generate_fig_3_2.py` (UQ Comparison)
   - `generate_fig_3_3.py` (Dataset Examples)

2. Install required packages:
```bash
pip install numpy matplotlib scipy
```

3. Run the scripts:
```bash
python generate_fig_3_2.py
python generate_fig_3_3.py
```

4. Include generated PDFs in your LaTeX document:
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_uq_comparison.pdf}
\caption{...}
\label{fig:uq_comparison}
\end{figure}
```

---

## Tips for Customization

- **Colors**: Adjust `fill=blue!20` in TikZ or `facecolor='blue'` in matplotlib
- **Sizes**: Modify `minimum width/height` in TikZ or `figsize` in matplotlib
- **Fonts**: Change font sizes with `fontsize=` parameter
- **Layout**: Adjust `node distance` in TikZ or subplot positions in matplotlib
- **DPI**: For higher quality, increase `dpi=300` to `dpi=600`

---

# Chapter 3 Figures

## Figure 3.1: Evidential FNO2d Architecture with Dimensions

### TikZ Code

```latex
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
    node distance=1.2cm and 0.8cm,
    layer/.style={rectangle, draw, minimum width=2.5cm, minimum height=1.2cm, align=center, font=\small},
    arrow/.style={->, >=stealth, thick}
]

% Input
\node[layer, fill=blue!20] (input) {
    \textbf{Input}\\
    $(B, 3, 64, 64)$
};

% Lifting layer
\node[layer, fill=green!20, right=of input] (lift) {
    \textbf{Lifting}\\
    $P: 3 \rightarrow 32$\\
    $(B, 32, 64, 64)$
};

% Fourier Layer 1
\node[layer, fill=orange!20, right=of lift] (f1) {
    \textbf{Fourier L1}\\
    modes=12\\
    $(B, 32, 64, 64)$
};

% Fourier Layer 2
\node[layer, fill=orange!20, right=of f1] (f2) {
    \textbf{Fourier L2}\\
    modes=12\\
    $(B, 32, 64, 64)$
};

% Fourier Layer 3
\node[layer, fill=orange!20, below=0.5cm of f2] (f3) {
    \textbf{Fourier L3}\\
    modes=12\\
    $(B, 32, 64, 64)$
};

% Fourier Layer 4
\node[layer, fill=orange!20, left=of f3] (f4) {
    \textbf{Fourier L4}\\
    modes=12\\
    $(B, 32, 64, 64)$
};

% Shared features before split
\node[layer, fill=purple!20, left=of f4] (shared) {
    \textbf{Features}\\
    $(B, 32, 64, 64)$
};

% Four output heads
\node[layer, fill=cyan!20, below=1.5cm of shared, xshift=-3cm] (gamma) {
    \textbf{Head 1}\\
    $Q_\gamma: 32 \rightarrow 1$\\
    $\gamma$\\
    $(B, 1, 64, 64)$
};

\node[layer, fill=cyan!20, right=0.5cm of gamma] (nu) {
    \textbf{Head 2}\\
    $Q_\nu: 32 \rightarrow 1$\\
    $\nu$\\
    $(B, 1, 64, 64)$
};

\node[layer, fill=cyan!20, right=0.5cm of nu] (alpha) {
    \textbf{Head 3}\\
    $Q_\alpha: 32 \rightarrow 1$\\
    $\alpha$\\
    $(B, 1, 64, 64)$
};

\node[layer, fill=cyan!20, right=0.5cm of alpha] (beta) {
    \textbf{Head 4}\\
    $Q_\beta: 32 \rightarrow 1$\\
    $\beta$\\
    $(B, 1, 64, 64)$
};

% Final concatenation
\node[layer, fill=red!20, below=1cm of nu] (output) {
    \textbf{Output}\\
    Concat($\gamma, \nu, \alpha, \beta$)\\
    $(B, 4, 64, 64)$
};

% Arrows - main flow
\draw[arrow] (input) -- (lift);
\draw[arrow] (lift) -- (f1);
\draw[arrow] (f1) -- (f2);
\draw[arrow] (f2) -- (f3);
\draw[arrow] (f3) -- (f4);
\draw[arrow] (f4) -- (shared);

% Skip connections
\draw[arrow, dashed, red, rounded corners] (f1.south) |- ++(0,-0.3) -| (f2.south);
\draw[arrow, dashed, red, rounded corners] (f2.south) |- ++(0,-0.3) -| (f3.south);
\draw[arrow, dashed, red, rounded corners] (f3.north) |- ++(0,0.3) -| (f4.north);

% Output head arrows
\draw[arrow] (shared.south) -- ++(0,-0.5) -| (gamma.north);
\draw[arrow] (shared.south) -- ++(0,-0.5) -| (nu.north);
\draw[arrow] (shared.south) -- ++(0,-0.5) -| (alpha.north);
\draw[arrow] (shared.south) -- ++(0,-0.5) -| (beta.north);

% Concatenation arrows
\draw[arrow] (gamma) -- (output);
\draw[arrow] (nu) -- (output);
\draw[arrow] (alpha) -- (output);
\draw[arrow] (beta) -- (output);

% Annotations
\node[above=0.3cm of f1, font=\small, text=orange] {width=32, layers=4};
\node[below=0.1cm of output, font=\tiny, text=red] {NIG parameters for uncertainty quantification};

\end{tikzpicture}
\caption{Evidential FNO2d architecture. The network takes input of shape $(B, 3, 64, 64)$, processes through a lifting layer to width 32, passes through 4 Fourier layers with 12 modes each, then splits into 4 parallel projection heads that output the NIG distribution parameters $(\gamma, \nu, \alpha, \beta)$, which are concatenated to form $(B, 4, 64, 64)$.}
\label{fig:evidential_fno_architecture}
\end{figure}
```

---

## Figure 3.2: Comparison of Three UQ Architectures

### Python Script

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def plot_uq_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Common parameters
    box_width = 0.15
    box_height = 0.08

    # ========== MC Dropout FNO ==========
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('(a) MC Dropout FNO', fontsize=14, fontweight='bold', pad=20)

    # Input
    input_box = FancyBboxPatch((0.4, 0.85), box_width, box_height,
                               boxstyle="round,pad=0.005",
                               edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(0.475, 0.89, 'Input\n(B,3,64,64)', ha='center', va='center', fontsize=9)

    # Lifting
    lift_box = FancyBboxPatch((0.4, 0.72), box_width, box_height,
                              boxstyle="round,pad=0.005",
                              edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(lift_box)
    ax.text(0.475, 0.76, 'Lifting\n(B,32,64,64)', ha='center', va='center', fontsize=9)

    # Fourier layers with dropout
    y_positions = [0.59, 0.46, 0.33, 0.20]
    for i, y in enumerate(y_positions):
        fourier_box = FancyBboxPatch((0.4, y), box_width, box_height,
                                     boxstyle="round,pad=0.005",
                                     edgecolor='black', facecolor='orange', linewidth=2)
        ax.add_patch(fourier_box)
        ax.text(0.475, y+0.04, f'Fourier L{i+1}\n+ Dropout', ha='center', va='center', fontsize=8)

        # Draw dropout mask icon
        for dx in [-0.02, 0, 0.02]:
            circle = Circle((0.65 + dx, y+0.04), 0.008, color='red', alpha=0.6)
            ax.add_patch(circle)

    # Projection
    proj_box = FancyBboxPatch((0.4, 0.07), box_width, box_height,
                              boxstyle="round,pad=0.005",
                              edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(proj_box)
    ax.text(0.475, 0.11, 'Projection\n(B,1,64,64)', ha='center', va='center', fontsize=9)

    # T=30 inference box
    inference_box = FancyBboxPatch((0.7, 0.35), 0.22, 0.15,
                                   boxstyle="round,pad=0.01",
                                   edgecolor='purple', facecolor='plum', linewidth=2, linestyle='--')
    ax.add_patch(inference_box)
    ax.text(0.81, 0.45, 'T=30 Forward', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.81, 0.40, 'Passes with', ha='center', va='center', fontsize=9)
    ax.text(0.81, 0.37, 'Different Masks', ha='center', va='center', fontsize=9)

    # Arrows
    for i in range(6):
        y_start = 0.85 - i * 0.13
        arrow = FancyArrowPatch((0.475, y_start - 0.04), (0.475, y_start - 0.09),
                               arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
        ax.add_patch(arrow)

    # ========== Deep Ensemble ==========
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('(b) Deep Ensemble FNO', fontsize=14, fontweight='bold', pad=20)

    # Draw 5 parallel models
    model_colors = ['lightblue', 'lightcoral', 'lightyellow', 'lightgreen', 'plum']
    x_positions = np.linspace(0.1, 0.75, 5)

    for idx, (x, color) in enumerate(zip(x_positions, model_colors)):
        # Input
        input_b = FancyBboxPatch((x, 0.85), 0.12, 0.06,
                                 boxstyle="round,pad=0.003",
                                 edgecolor='black', facecolor=color, linewidth=1.5, alpha=0.7)
        ax.add_patch(input_b)
        ax.text(x+0.06, 0.88, f'M{idx+1}', ha='center', va='center', fontsize=8, fontweight='bold')

        # FNO stack (simplified)
        fno_b = FancyBboxPatch((x, 0.50), 0.12, 0.30,
                               boxstyle="round,pad=0.003",
                               edgecolor='black', facecolor=color, linewidth=1.5, alpha=0.5)
        ax.add_patch(fno_b)
        ax.text(x+0.06, 0.65, 'FNO', ha='center', va='center', fontsize=9, fontweight='bold')

        # Output
        out_b = FancyBboxPatch((x, 0.35), 0.12, 0.06,
                               boxstyle="round,pad=0.003",
                               edgecolor='black', facecolor=color, linewidth=1.5, alpha=0.7)
        ax.add_patch(out_b)
        ax.text(x+0.06, 0.38, f'$\hat{{y}}_{idx+1}$', ha='center', va='center', fontsize=8)

        # Arrows
        arrow1 = FancyArrowPatch((x+0.06, 0.85), (x+0.06, 0.80),
                                arrowstyle='->', mutation_scale=10, linewidth=1.5, color='black')
        ax.add_patch(arrow1)

        arrow2 = FancyArrowPatch((x+0.06, 0.50), (x+0.06, 0.41),
                                arrowstyle='->', mutation_scale=10, linewidth=1.5, color='black')
        ax.add_patch(arrow2)

    # Aggregation
    agg_box = FancyBboxPatch((0.35, 0.15), 0.20, 0.12,
                             boxstyle="round,pad=0.01",
                             edgecolor='purple', facecolor='lightgray', linewidth=2)
    ax.add_patch(agg_box)
    ax.text(0.45, 0.23, 'Aggregate', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.45, 0.19, 'Mean & Variance', ha='center', va='center', fontsize=9)

    # Arrows to aggregation
    for x in x_positions:
        arrow = FancyArrowPatch((x+0.06, 0.35), (0.45, 0.27),
                               arrowstyle='->', mutation_scale=10, linewidth=1,
                               color='purple', linestyle='--', alpha=0.6)
        ax.add_patch(arrow)

    ax.text(0.45, 0.05, 'M=5 Independent Models', ha='center', va='center',
            fontsize=9, style='italic', color='red')

    # ========== Evidential FNO ==========
    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('(c) Evidential FNO', fontsize=14, fontweight='bold', pad=20)

    # Input
    input_box = FancyBboxPatch((0.4, 0.85), box_width, box_height,
                               boxstyle="round,pad=0.005",
                               edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(0.475, 0.89, 'Input\n(B,3,64,64)', ha='center', va='center', fontsize=9)

    # Lifting
    lift_box = FancyBboxPatch((0.4, 0.72), box_width, box_height,
                              boxstyle="round,pad=0.005",
                              edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(lift_box)
    ax.text(0.475, 0.76, 'Lifting\n(B,32,64,64)', ha='center', va='center', fontsize=9)

    # Fourier layers (no dropout)
    y_positions = [0.59, 0.46, 0.33, 0.20]
    for i, y in enumerate(y_positions):
        fourier_box = FancyBboxPatch((0.4, y), box_width, box_height,
                                     boxstyle="round,pad=0.005",
                                     edgecolor='black', facecolor='orange', linewidth=2)
        ax.add_patch(fourier_box)
        ax.text(0.475, y+0.04, f'Fourier L{i+1}', ha='center', va='center', fontsize=9)

    # Four output heads
    head_names = ['γ', 'ν', 'α', 'β']
    head_colors = ['cyan', 'lime', 'yellow', 'pink']
    x_heads = [0.25, 0.37, 0.49, 0.61]

    for i, (x_h, name, color) in enumerate(zip(x_heads, head_names, head_colors)):
        head_box = FancyBboxPatch((x_h, 0.05), 0.10, 0.08,
                                  boxstyle="round,pad=0.005",
                                  edgecolor='black', facecolor=color, linewidth=2, alpha=0.6)
        ax.add_patch(head_box)
        ax.text(x_h+0.05, 0.09, f'Head\n${name}$', ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrow from last Fourier layer to head
        arrow = FancyArrowPatch((0.475, 0.20), (x_h+0.05, 0.13),
                               arrowstyle='->', mutation_scale=12, linewidth=1.5, color='purple')
        ax.add_patch(arrow)

    # NIG distribution box
    nig_box = FancyBboxPatch((0.75, 0.35), 0.20, 0.20,
                             boxstyle="round,pad=0.01",
                             edgecolor='purple', facecolor='plum', linewidth=2, linestyle='--')
    ax.add_patch(nig_box)
    ax.text(0.85, 0.50, 'NIG', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(0.85, 0.46, 'Distribution', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(0.85, 0.42, 'Single Forward', ha='center', va='center', fontsize=9)
    ax.text(0.85, 0.38, 'Pass', ha='center', va='center', fontsize=9)

    # Arrows
    for i in range(6):
        y_start = 0.85 - i * 0.13
        arrow = FancyArrowPatch((0.475, y_start - 0.04), (0.475, y_start - 0.09),
                               arrowstyle='->', mutation_scale=15, linewidth=2, color='black')
        ax.add_patch(arrow)

    plt.tight_layout()
    plt.savefig('fig_uq_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fig_uq_comparison.png', bbox_inches='tight', dpi=300)
    print("Figure 3.2 saved as fig_uq_comparison.pdf and .png")
    plt.show()

if __name__ == '__main__':
    plot_uq_comparison()
```

---

## Figure 3.3: Dataset Examples

### Python Script

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_dataset_examples():
    """
    Generate example visualizations for NS and Darcy datasets.
    Uses synthetic data for demonstration - replace with actual data if available.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Grid resolution
    n = 64
    x = np.linspace(0, 2*np.pi, n)
    y = np.linspace(0, 2*np.pi, n)
    X, Y = np.meshgrid(x, y)

    # ========== (a) NS Vorticity Field ==========
    ax = axes[0, 0]

    # Simulate vorticity field (similar to NS turbulence)
    np.random.seed(42)
    vorticity = np.zeros((n, n))
    for k in range(1, 8):
        for l in range(1, 8):
            phase = np.random.rand() * 2 * np.pi
            amplitude = np.random.randn() / (k**2 + l**2)
            vorticity += amplitude * np.sin(k*X + l*Y + phase)

    im1 = ax.imshow(vorticity, cmap='RdBu_r', extent=[0, 2*np.pi, 0, 2*np.pi],
                    origin='lower', vmin=-2, vmax=2)
    ax.set_title('(a) Navier-Stokes Vorticity Field', fontsize=12, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046)
    cbar1.set_label('$\omega(x,y)$', fontsize=10)
    ax.text(0.05, 0.95, f'Resolution: {n}×{n}', transform=ax.transAxes,
            fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ========== (b) NS Velocity Magnitude ==========
    ax = axes[0, 1]

    # Velocity magnitude (derived from vorticity)
    # Simulate velocity field
    u = np.zeros((n, n))
    v = np.zeros((n, n))
    for k in range(1, 6):
        for l in range(1, 6):
            phase_u = np.random.rand() * 2 * np.pi
            phase_v = np.random.rand() * 2 * np.pi
            amp = 1.0 / (k + l)
            u += amp * np.sin(k*X + phase_u)
            v += amp * np.cos(l*Y + phase_v)

    velocity_magnitude = np.sqrt(u**2 + v**2)

    im2 = ax.imshow(velocity_magnitude, cmap='viridis', extent=[0, 2*np.pi, 0, 2*np.pi],
                    origin='lower')
    ax.set_title('(b) Navier-Stokes Velocity Magnitude', fontsize=12, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046)
    cbar2.set_label('$|\\mathbf{u}(x,y)|$', fontsize=10)
    ax.text(0.05, 0.95, f'Resolution: {n}×{n}', transform=ax.transAxes,
            fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add velocity field arrows
    stride = 4
    ax.quiver(X[::stride, ::stride], Y[::stride, ::stride],
              u[::stride, ::stride], v[::stride, ::stride],
              alpha=0.4, color='white', scale=20)

    # ========== (c) Darcy Permeability ==========
    ax = axes[1, 0]

    # Simulate permeability field (log-normal)
    np.random.seed(123)
    permeability = np.zeros((n, n))
    for k in range(1, 10):
        for l in range(1, 10):
            phase = np.random.rand() * 2 * np.pi
            amplitude = np.random.randn() * 0.5 / (k + l)
            permeability += amplitude * np.sin(k*X + l*Y + phase)

    # Make it log-normal
    permeability = np.exp(permeability)

    im3 = ax.imshow(permeability, cmap='YlOrBr', extent=[0, 1, 0, 1],
                    origin='lower', vmin=0, vmax=5)
    ax.set_title('(c) Darcy Permeability Field', fontsize=12, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    cbar3 = plt.colorbar(im3, ax=ax, fraction=0.046)
    cbar3.set_label('$K(x,y)$', fontsize=10)
    ax.text(0.05, 0.95, f'Resolution: {n}×{n}', transform=ax.transAxes,
            fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ========== (d) Darcy Pressure ==========
    ax = axes[1, 1]

    # Simulate pressure solution (smoother than permeability)
    pressure = np.zeros((n, n))
    for k in range(1, 5):
        for l in range(1, 5):
            phase = np.random.rand() * 2 * np.pi
            amplitude = np.random.randn() / (k**2 + l**2 + 1)
            pressure += amplitude * np.cos(k*X*np.pi + l*Y*np.pi + phase)

    # Add boundary gradient
    pressure += 1.0 - 0.5 * X / (2*np.pi)

    im4 = ax.imshow(pressure, cmap='coolwarm', extent=[0, 1, 0, 1],
                    origin='lower')
    ax.set_title('(d) Darcy Pressure Field', fontsize=12, fontweight='bold')
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    cbar4 = plt.colorbar(im4, ax=ax, fraction=0.046)
    cbar4.set_label('$p(x,y)$', fontsize=10)
    ax.text(0.05, 0.95, f'Resolution: {n}×{n}', transform=ax.transAxes,
            fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Add contour lines
    contours = ax.contour(X/(2*np.pi), Y/(2*np.pi), pressure, levels=10,
                          colors='black', alpha=0.3, linewidths=0.5)

    plt.tight_layout()
    plt.savefig('fig_dataset_examples.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('fig_dataset_examples.png', bbox_inches='tight', dpi=300)
    print("Figure 3.3 saved as fig_dataset_examples.pdf and .png")
    plt.show()

if __name__ == '__main__':
    plot_dataset_examples()
```

---

## Updated Usage Instructions for Chapter 3

### Generate Chapter 3 Figures:

1. **Figure 3.1 (TikZ)**: Copy the TikZ code into your LaTeX document

2. **Figures 3.2 and 3.3 (Python)**:
```bash
# Extract scripts from img.md and save as:
# - generate_fig_3_2.py
# - generate_fig_3_3.py

python generate_fig_3_2.py
python generate_fig_3_3.py
```

3. **Include in LaTeX**:
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig_uq_comparison.pdf}
\caption{Comparison of three uncertainty quantification architectures...}
\label{fig:uq_comparison}
\end{figure}
```