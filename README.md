QuadProg
========

[![Crate](https://img.shields.io/crates/v/quadprog.svg)](https://crates.io/crates/quadprog)
[![Build](https://github.com/erikbrinkman/quadprog-rs/actions/workflows/build.yml/badge.svg)](https://github.com/erikbrinkman/quadprog-rs/actions/workflows/build.yml)

A dense quadratic program solver in pure rust, based on the Goldfarb Idnani algorithm.
This implementation is based on a few sources, but drew primarily from [quadprog](https://github.com/quadprog/quadprog).

Usage
-----

Add this to your `Cargo.toml`:

```
[dependencies]
quadprog = "0.0.1"
```

Then solve using:
```
quadprog::solve_qp(...)
```
