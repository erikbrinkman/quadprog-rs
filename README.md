quadprog
========

[![Crate](https://img.shields.io/crates/v/quadprog.svg)](https://crates.io/crates/quadprog)
[![Docs](https://docs.rs/quadprog/badge.svg)](https://docs.rs/quadprog)
[![Build](https://github.com/erikbrinkman/quadprog-rs/actions/workflows/build.yml/badge.svg)](https://github.com/erikbrinkman/quadprog-rs/actions/workflows/build.yml)

A dense quadratic program solver in pure rust, based on the Goldfarb-Idnani algorithm.
This implementation draws primarily from [quadprog](https://github.com/quadprog/quadprog).

Usage
-----

Add this to your `Cargo.toml`:

```toml
[dependencies]
quadprog = "*"
```

Then solve a strictly convex QP of the form:

```text
    minimize     1/2 x' Q x + c' x
    subject to   A1 x  = b1
                 A2 x <= b2
```

For example, to solve

```text
    minimize     1/2 x^2 + 1/2 y^2 + x
    subject to   x + 2 y >= 1
```

write

```rust
use quadprog::solve_qp;
let mut q = [1., 0., 0., 1.];
let c = [1., 0.];
let a = [-1., -2.];
let b = [-1.];
let sol = solve_qp(&mut q, &c, &a, &b, 0, false).unwrap();
assert_eq!(sol.sol, &[-0.6, 0.8]);
```

References
----------

D. Goldfarb and A. Idnani (1983). A numerically stable dual method for solving strictly convex quadratic programs. *Mathematical Programming*, 27, 1-33.
