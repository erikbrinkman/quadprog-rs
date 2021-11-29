QuadProg
========

A dense quadratic program solver in pure rust, based on the Goldfarb Idnani algorithm.
This implementation is based on a few sources, but drew primarily from [quadprog](https://github.com/quadprog/quadprog).

Usage
-----

Add this to your `Cargo.toml`:

```
[dependencies]
quadprog = "0.0.0"
```

Then solve using:
```
quadprog::solve_qp(...)
```
