# JSINDy

JSINDy (Joint SINDy) is an approach to jointly learning sparse governing equations and system states from limited, incomplete, and noisy observations. JSINDy combines sparse recovery strategies over a function library, a least squares collocation approach to solving ODEs, and reproducing kernel Hilbert space (RKHS) regularization to simultaneously fit dynamics and estimate trajectories.

**Paper:** [A joint optimization approach to identifying sparse dynamics using least squares kernel collocation](https://arxiv.org/abs/2511.18555)

## Installation

Requires Python >= 3.10, and you may want to manually install ```jax[cuda12]``` first before ```pip install -e .```

## Citation

A. W. Hsu, I. W. Griss Salas, J. M. Stevens-Haas, J. N. Kutz, A. Aravkin, and B. Hosseini, "A joint optimization approach to identifying sparse dynamics using least squares kernel collocation," arXiv preprint arXiv:2511.18555, 2025.

```bibtex
@misc{hsu2025jointoptimizationapproachidentifying,
      title={A joint optimization approach to identifying sparse dynamics using least squares kernel collocation},
      author={Alexander W. Hsu and Ike W. Griss Salas and Jacob M. Stevens-Haas and J. Nathan Kutz and Aleksandr Aravkin and Bamdad Hosseini},
      year={2025},
      eprint={2511.18555},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2511.18555},
}
```

## License

MIT
