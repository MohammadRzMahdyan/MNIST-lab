## MNIST Dataset | Prediction by Fully Connected Neural Networks
---
### شبکه های عصبی آموزش دیده روی دیتا ست `MNIST`
- شامل دولایه پنهان
$$
\begin{aligned}
hidden_1 : 128D \\
hidden_2 : 64D\\
output_{softmax}: 10\\
\end{aligned}
$$
$784D\rightarrow128D\rightarrow64\rightarrow10_{softmax}
$$
---
structure:
```
.
├── dataset
│   └── mnist.npz `functions`
├── notebook
│   └── main.ipynb 'run`
├── README.md
├── requirements.txt
└── src
	├── func.py

```
---
#tensorflow #keras #numpy
