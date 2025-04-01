# Upper Bound Analysis


<div align=center>
<img src="https://github.com/XXTEST1XX/BOUND-ANALYSIS/blob/main/upper_bounds.png" width="50%" height="50%" />
</div>

**<p align="center">CSWin-T [1,2]</p>**


Fig. R2-1 The upper bound of different attention mechanisms (EFFATT [3], ELFATT, and LOCAL [1]) for approximating attention matrices of vanilla attention during the training of ImageNet-1K. The backbone used is CSWin-T [1,2]. The upper bound for ELFATT is obtained according to 

$$
\left\lVert\left[ \mathrm{exp}(\bar{\mathbf{Q}})\mathrm{exp}(\bar{\mathbf{K}})^{\top},\ \left( \mathrm{exp}(\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^{\top})\odot\mathbf{Z}\right)\right] - \left[ \mathrm{exp}(\bar{\mathbf{Q}}\bar{\mathbf{K}}^{\top}),\ \mathrm{exp}(\tilde{\mathbf{Q}}\tilde{\mathbf{K}}^{\top})\right] \right\rVert_{\xi}
\leq
\left\lVert \mathrm{exp}(\bar{\mathbf{Q}})\mathrm{exp}(\bar{\mathbf{K}})^{\top}-\mathrm{exp}(\bar{\mathbf{Q}} \bar{\mathbf{K}}^{\top}) \right\rVert_{\xi}
+
\left\lVert\mathrm{exp}(\tilde{\mathbf{Q}}\tilde{\mathbf{K}}^{\top})\odot\mathbf{Z}-\mathrm{exp}(\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^{\top})\right\rVert_{\xi},
$$ 


the upper bound for EFFATT is as follows,

$$
\left\lVert\left[ \mathrm{exp}(\bar{\mathbf{Q}})\mathrm{exp}(\bar{\mathbf{K}})^{\top},\ \mathrm{exp}(\tilde{\mathbf{Q}}) \mathrm{exp}(\tilde{\mathbf{K}})^{\top}\right] - \left[ \mathrm{exp}(\bar{\mathbf{Q}}\bar{\mathbf{K}}^{\top}),\ \mathrm{exp}(\tilde{\mathbf{Q}}\tilde{\mathbf{K}}^{\top})\right] \right\rVert_{\xi}
\leq
\left\lVert \mathrm{exp}(\bar{\mathbf{Q}})\mathrm{exp}(\bar{\mathbf{K}})^{\top}-\mathrm{exp}(\bar{\mathbf{Q}} \bar{\mathbf{K}}^{\top}) \right\rVert_{\xi}
+
\left\lVert\mathrm{exp}(\tilde{\mathbf{Q}})\mathrm{exp}(\tilde{\mathbf{K}})^{\top}-\mathrm{exp}(\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^{\top})\right\rVert_{\xi},
$$ 

and for LOCAL is as follows,

$$
\left\lVert\left[\left(\mathrm{exp}(\bar{\mathbf{Q}} \bar{\mathbf{K}}^{\top})\odot\mathbf{Z_1}\right),\ \left(\mathrm{exp}(\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^{\top})\odot\mathbf{Z_2}\right)\right] - \left[\mathrm{exp}(\bar{\mathbf{Q}}\bar{\mathbf{K}}^{\top}),\ \mathrm{exp}(\tilde{\mathbf{Q}}\tilde{\mathbf{K}}^{\top})\right] \right\rVert_{\xi}
\leq
\left\lVert\mathrm{exp}(\bar{\mathbf{Q}} \bar{\mathbf{K}}^{\top})\odot\mathbf{Z_1}-\mathrm{exp}(\bar{\mathbf{Q}} \bar{\mathbf{K}}^{\top}) \right\rVert_{\xi}
+
\left\lVert\mathrm{exp}(\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^{\top})\odot\mathbf{Z_2}-\mathrm{exp}(\tilde{\mathbf{Q}} \tilde{\mathbf{K}}^{\top})\right\rVert_{\xi},
$$ 

where $\xi=F$ denotes the Frobenius norm, and $\mathbf{Z}_1$ and $\mathbf{Z}_2$ are different to $\mathbf{Z}$ of ELFATT, because LOCAL uses a more complex cross-shaped blockify method for different heads. All upper bounds were obtained from the final attention layer of the second level of the backbone. The upper bounds of different methods show a decreasing trend as the training process progresses. ELFATT further reduces the upper bound of the LOCAL attention mechanism, although it still has a higher bound than EFFATT, which is caused by its sparse blockify attention heads.


<div align=center>
<img src="https://github.com/XXTEST1XX/BOUND-ANALYSIS/blob/main/global_attention_heads.png" width="50%" height="50%" />
</div>

**<p align="center">(a) Global Linear Attention Heads</p>**

<div align=center>
<img src="https://github.com/XXTEST1XX/BOUND-ANALYSIS/blob/main/sparse_attention_heads.png" width="50%" height="50%" />
</div>

**<p align="center">(b) Sparse Blockify Attention Heads</p>**

Fig. R2-2 The comparison of relative attention matrix approximation error ($\left\lVert\mathbf{A}-\mathbf{A}^{'}\right\rVert_{\xi}/\left\lVert\mathbf{A}\right\rVert_{\xi}$, where $\mathbf{A}$ denotes the attention matrix of VaniATT and $\textit{\textbf{A}}'$ denotes the attention matrix of linear attention) of global linear attention heads of ELFATT and the corresponding heads in EFFATT and relative attention matrix approximation error of sparse blockify attention heads of ELFATT and the corresponding heads in LOCAL. The backbone used is CSWin-T. Since ELFATT uses a hybrid head architecture, its half heads consist of global linear attention heads, and the remaining heads consist of sparse blockify attention heads. We compared the relative attention matrix approximation error of global linear attention heads of ELFATT and the heads in the same position of EFFATT. We compared the relative attention matrix approximation error of sparse blockify attention heads of ELFATT and the heads in the same position of LOCAL. We found that the global linear attention heads of ELFATT have a lower approximation error than the heads in the same position of EFFATT, and the sparse blockify attention heads of ELFATT have a lower approximation error than the heads in the same position of LOCAL. The hybrid head architecture has complementary effects on approximation error reduction.

**References**:

[1] X. Dong, J. Bao, D. Chen, W. Zhang, N. Yu, L. Yuan, D. Chen, and B. Guo, “CSWin Transformer: A general vision Transformer backbone with cross-shaped windows,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 12124–12134.

[2] D. Han, X. Pan, Y. Han, S. Song, and G. Huang, “FLatten Transformer: Vision Transformer using focused linear attention,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 5961–5971.

[3] Z. Shen, M. Zhang, H. Zhao, S. Yi, and H. Li, “Efficient Attention: Attention with linear complexities,” in *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2021, pp. 3531–3539.
