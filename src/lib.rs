//! Solve dense quadratic programs.
//!
//! This crate implements the Goldfarb Indiani method[^1] for solving quadratic programs of the
//! form:
//!
//! ```text
//!     minimize     1/2 x' Q x + c' x
//!     subject to   A1 x  = b1
//!                  A2 x <= b2
//! ```
//!
//! in pure rust. These are solved via the only exported function [solve_qp] which returns a
//! [Solution] struct.
//!
//! # Examples
//!
//! If we want to solve
//! ```text
//!     minimize     1/2 x^2 + 1/2 y^2 + x
//!     subject to   x + 2 y >= 1
//! ```
//!
//! we can do so with the following example:
//!
//! ```
//! # use quadprog::solve_qp;
//! let mut q = [1., 0., 0., 1.];
//! let c = [1., 0.];
//! let a = [-1., -2.];
//! let b = [-1.];
//! let sol = solve_qp(&mut q, &c, &a, &b, 0, false).unwrap();
//! assert_eq!(sol.sol, &[-0.6, 0.8]);
//! ```
//!
//! [^1] D. Goldfarb and A. Idnani (1983). A numerically stable dual
//!     method for solving strictly convex quadratic programs.
//!     Mathematical Programming, 27, 1-33.

#[cfg(test)]
extern crate approx;

use std::cmp::min;

/// integer sqrt
fn usqrt(val: usize) -> usize {
    (val as f64).sqrt() as usize
}

/// y = a * x + y
fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// dot product of two vectors
fn dot(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let mut result = 0.0;
    for (xi, yi) in x.iter().zip(y) {
        result += xi * yi;
    }
    result
}

/// scale a vector
fn scal(mult: f64, vec: &mut [f64]) {
    for val in vec {
        *val *= mult;
    }
}

/// Compute a * b, where a is upper triangular.
/// The result is written into b.
fn triangular_multiply(mat: &[f64], vec: &mut [f64]) {
    let n = vec.len();
    debug_assert_eq!(mat.len(), n * n);
    for j in 0..n {
        axpy(vec[j], &mat[j * n..j * n + j], &mut vec[..j]);
        vec[j] *= mat[j + j * n];
    }
}

/// Compute transpose(a) * b, where a is upper triangular.
/// The result is written into b.
fn triangular_multiply_transpose(mat: &[f64], vec: &mut [f64]) {
    let n = vec.len();
    debug_assert_eq!(mat.len(), n * n);
    for j in (0..n).rev() {
        vec[j] *= mat[j + j * n];
        vec[j] += dot(&vec[..j], &mat[j * n..j * n + j]);
    }
}

/// Solve a * x = b, where a is upper triangular.
/// The solution is written into b.
fn triangular_solve(mat: &[f64], vec: &mut [f64]) {
    let n = vec.len();
    debug_assert_eq!(mat.len(), n * n);
    for k in (0..n).rev() {
        vec[k] /= mat[k + k * n];
        axpy(-vec[k], &mat[k * n..k * n + k], &mut vec[..k]);
    }
}

/// Solve transpose(a) * x = b, where a is upper triangular.
/// The solution is written into b.
fn triangular_solve_transpose(mat: &[f64], vec: &mut [f64]) {
    let n = vec.len();
    debug_assert_eq!(mat.len(), n * n);
    for k in 0..n {
        vec[k] -= dot(&mat[k * n..k * n + k], &vec[..k]);
        vec[k] /= mat[k + k * n];
    }
}

/// Invert a, where a is upper triangular.
/// The inverse is written into a.
fn triangular_invert(mat: &mut [f64]) {
    let n = usqrt(mat.len());
    debug_assert_eq!(mat.len(), n * n);
    for k in 0..n {
        mat[k + k * n] = 1.0 / mat[k + k * n];
        scal(-mat[k + k * n], &mut mat[k * n..k * n + k]);

        let (left, right) = mat.split_at_mut(n + k * n);

        for j in 0..n - k - 1 {
            axpy(
                right[k + j * n],
                &left[k * n..k * n + k],
                &mut right[j * n..j * n + k],
            );
            right[k + j * n] *= left[k + k * n];
        }
    }
}

/// Find the upper triangular matrix r such that a = transpose(r) * r, where a is positive definite.
/// The result is written into the upper triangle of a.
/// Errs if a is not positive definite.
fn cholesky(mat: &mut [f64]) -> Result<(), &'static str> {
    let n = usqrt(mat.len());
    debug_assert_eq!(n * n, mat.len());
    for j in 0..n {
        for k in 0..j {
            mat[k + j * n] = (mat[k + j * n] - dot(&mat[k * n..k * n + k], &mat[j * n..j * n + k]))
                / mat[k + k * n];
        }

        let s = mat[j + j * n] - dot(&mat[j * n..j * n + j], &mat[j * n..j * n + j]);
        if s <= 0.0 {
            return Err("matrix not positive definite");
        }

        mat[j + j * n] = s.sqrt();
    }

    Ok(())
}

/// compute sqrt(a^2 + b^2)
fn hypot(left: f64, right: f64) -> f64 {
    let (min, max) = if left < right {
        (left, right)
    } else {
        (right, left)
    };
    let ratio = min / max;
    max * (1.0 + ratio * ratio).sqrt()
}

/// get length len slices to the left and right of split
///
/// for a matrix this will be neighboring rows.
fn left_right_slices<'a, T>(
    slice: &'a mut [T],
    split: usize,
    len: usize,
) -> (&'a mut [T], &'a mut [T]) {
    let (left, right) = slice.split_at_mut(split);
    (&mut left[split - len..], &mut right[..len])
}

/// Apply orthogonal transformations to `vec` to bring the components beyond the rth to zero.
/// Apply the same orthogonal transformations to the columns of `mat`.
///
/// Note: the trailing elements of `vec` are not actually zeroed out
fn qr_insert(r: usize, vec: &mut [f64], mat: &mut [f64]) {
    let n = vec.len();
    debug_assert_eq!(mat.len(), n * n);
    debug_assert!(r <= n);
    for i in (r..n).rev() {
        // On this iteration, reduce a[i] to zero.
        if vec[i] == 0.0 {
            continue;
        }

        let (left, right) = left_right_slices(mat, i * n, n);
        if vec[i - 1] == 0.0 {
            // Simply swap
            vec[i - 1] = vec[i];
            left.swap_with_slice(right);
        } else {
            // Compute a Givens rotation.
            let h = hypot(vec[i - 1], vec[i]).copysign(vec[i - 1]);
            let gc = vec[i - 1] / h;
            let gs = vec[i] / h;
            // this saves a fourth multiplication in the inner loop below
            let nu = vec[i] / (vec[i - 1] + h);
            vec[i - 1] = h;

            for (li, ri) in left.iter_mut().zip(right.iter_mut()) {
                let temp = gc * *li + gs * *ri;
                *ri = nu * (*li + temp) - *ri;
                *li = temp;
            }
        }
    }
}

/// Drop the col-th column of rmat.
/// Apply orthogonal transformations to the rows of rmat to restore rmat to upper triangular form.
/// Apply the same orthogonal transformations to the columns of qmat.
///
/// Note that rmat is an r by r upper triangular matrix stored as packed columns.
/// So the input size is r*(r+1)/2 and the output size is (r-1)*r/2.
fn qr_delete(col: usize, qmat: &mut [f64], rmat: &mut [f64]) {
    let n = usqrt(qmat.len());
    debug_assert_eq!(qmat.len(), n * n);
    let r = (usqrt(8 * rmat.len() + 1) - 1) / 2;
    debug_assert_eq!(r * (r + 1) / 2, rmat.len());

    for i in col..r {
        // On this iteration, reduce the (i+1, i+1) element of R to zero,
        // and then move column (i+1) to position i.

        // R[l] is the (i+1, i+1) element of R.
        let di = i * (i + 1) / 2;
        let l = di + i;
        if rmat[l] == 0.0 {
            continue;
        }

        let (left, right) = left_right_slices(qmat, i * n, n);
        if rmat[l - 1] == 0.0 {
            // Simply swap.
            let mut ind = l;
            for j in i + 1..=r {
                rmat.swap(ind - 1, ind);
                ind += j;
            }

            left.swap_with_slice(right);
        } else {
            // Compute a Givens rotation.
            let h = hypot(rmat[l - 1], rmat[l]).copysign(rmat[l - 1]);
            let gc = rmat[l - 1] / h;
            let gs = rmat[l] / h;
            // this saves a fourth multiplication in the inner loop below
            let nu = rmat[l] / (rmat[l - 1] + h);

            let mut ind = l;
            for j in i + 1..=r {
                let temp = gc * rmat[ind - 1] + gs * rmat[ind];
                rmat[ind] = nu * (rmat[ind - 1] + temp) - rmat[ind];
                rmat[ind - 1] = temp;
                ind += j;
            }

            for (li, ri) in left.iter_mut().zip(right.iter_mut()) {
                let temp = gc * *li + gs * *ri;
                *ri = nu * (*li + temp) - *ri;
                *li = temp;
            }
        }

        let (left, right) = left_right_slices(rmat, di, i);
        left.swap_with_slice(right);
    }
}

/// The solution to a quadratic program
///
/// - `obj` is the objective value
/// - `sol` is the vector solution
/// - `lagr` is a vector the lagrange multipliers for each constraint
/// - `iact` are the indices of the active constraints
/// - `iter` is the number of added constraints over the course of execution
#[derive(Debug)]
pub struct Solution {
    pub obj: f64,
    pub sol: Vec<f64>,
    pub lagr: Vec<f64>,
    pub iact: Vec<usize>,
    pub iter: usize,
}

/// Solve a strictly convex quadratic program.
///
/// The program takes the form of:
///
/// ```text
///     minimize     1/2 x' Q x + c' x
///     subject to   A1 x  = b1
///                  A2 x <= b2
/// ```
///
/// where `A1` is the first `meq` rows of `amat`, and `b1` is the first `meq` elements of `bvec`.
/// Both `qmat` and `amat` must be in row-major order, e.g. the first elements of amat correspond
/// to its first row.
///
/// If `factorized` is true, qmat should be the upper triangular cholesky decomposition of `Q`, i.e.
/// `Q` should instead be `L'` in `L L' = Q`.
///
/// Note that `Q` is mutable, and is used for part of the computation. If you need to use `Q`
/// afterward, make a copy first.
pub fn solve_qp(
    qmat: &mut [f64],
    cvec: &[f64],
    amat: &[f64],
    bvec: &[f64],
    meq: usize,
    factorized: bool,
) -> Result<Solution, &'static str> {
    let n = cvec.len();
    let q = bvec.len();
    let r = min(n, q);
    if qmat.len() != n * n {
        return Err("qmat was not appropriate size given cvec");
    }
    if amat.len() != n * q {
        return Err("amat was not appropriate size given cvec and bvec");
    }

    // NOTE we allocate all the work space in one go to remove unnecessary applications
    let mut work = vec![0.0; n + n + r + r * (r + 1) / 2 + q + q];
    let (dv, rest) = work.split_at_mut(n);
    let (zv, rest) = rest.split_at_mut(n);
    let (rv_mem, rest) = rest.split_at_mut(r);
    let (rmat, rest) = rest.split_at_mut(r * (r + 1) / 2);
    let (sv, nbv) = rest.split_at_mut(q);

    let mut iact: Vec<usize> = Vec::with_capacity(r);
    let mut uv: Vec<f64> = Vec::with_capacity(r);

    let mut sol = cvec.to_owned();
    scal(-1.0, &mut sol);
    if factorized {
        // G is already L^-T
        triangular_multiply_transpose(qmat, &mut sol);
        // now sol contains L^-1 a
        triangular_multiply(qmat, &mut sol);
        // now sol contains L^-T L^-1 a = G^-1 a
    } else {
        cholesky(qmat)?;
        // now the upper triangle of G contains L^T
        triangular_solve_transpose(qmat, &mut sol);
        // now sol contains L^-1 a
        triangular_solve(qmat, &mut sol);
        // now sol contains L^-T L^-1 a = G^-1 a
        triangular_invert(qmat);
        // now G contains L^-T
    }

    // Set the lower triangle of J to zero.
    for j in 0..n {
        for i in j + 1..n {
            qmat[i + j * n] = 0.0;
        }
    }

    // Calculate the objective value at the unconstrained minimum.
    let mut obj = dot(cvec, &sol) / 2.0;

    // Calculate the norm of each column of the C matrix.
    // This will be used in our pivoting rule.
    for (nbvi, arow) in nbv.iter_mut().zip(amat.chunks_exact(n)) {
        *nbvi = dot(arow, arow).sqrt();
    }

    let mut iadd;
    let mut iter = 0_usize;

    while {
        // NOTE Here we store the slack variables so that we can efficiently zero out active
        // constraints incase of numerical imprecision. To avoid this memoty we'd have to either
        // hash the indices (still extra  or suffer quadratic running time.

        // Calculate the slack variables C^T xv - bv and store the result in sv.
        for ((arow, bvi), svi) in amat.chunks_exact(n).zip(bvec).zip(sv.iter_mut()) {
            *svi = bvi - dot(&sol, arow);
        }

        // Force the slack variables to zero for constraints in the active set,
        // as a safeguard against rounding errors.
        for ind in iact.iter().copied() {
            sv[ind] = 0.0;
        }

        // Choose a violated constraint to add to the active set.
        // We choose the constraint with the largest violation.
        // The index of the constraint to add is stored in iadd.
        iadd = q;
        let mut max_violation = 0.0;
        for (i, (svi, nbvi)) in sv.iter().zip(nbv.iter()).enumerate() {
            if *svi < -max_violation * nbvi - f64::EPSILON {
                iadd = i;
                max_violation = -svi / nbvi;
            } else if i < meq && *svi > max_violation * nbvi + f64::EPSILON {
                iadd = i;
                max_violation = svi / nbvi;
            }
        }

        // not ll constraints are satisfied, so we need to continue
        iadd != q
    } {
        iter += 1;

        let aadd = amat.chunks_exact(n).nth(iadd).unwrap();
        let mut slack = sv[iadd];
        let mut u = 0.0;
        let direc = slack.signum();

        let mut idel: usize; // the constraint to remove from the active set
        while {
            // Set dv = J^T n, where n is the column of C corresponding to the constraint
            // that we are adding to the active set.
            for (dvi, grow) in dv.iter_mut().zip(qmat.chunks_exact(n)) {
                *dvi = direc * dot(grow, aadd);
            }

            // Set zv = J_2 d_2. This is the step direction for the primal variable xv.
            zv.fill(0.0);
            for (grow, dvi) in qmat.chunks_exact(n).zip(dv.iter()).skip(iact.len()) {
                axpy(*dvi, grow, zv);
            }

            // Set rv = R^-1 d_1. This is (the negative of) the step direction for the dual variable uv.
            let rv = &mut rv_mem[..iact.len()];
            rv.clone_from_slice(&dv[..iact.len()]);
            for i in (0..iact.len()).rev() {
                let start = i * (i + 1) / 2;
                rv[i] /= rmat[start + i];
                axpy(-rv[i], &rmat[start..start + i], &mut rv[..i]);
            }

            // Find the largest step length t1 before dual feasibility is violated.
            // Store in idel the index of the constraint to remove from the active set, if we get
            // that far.
            idel = r;
            let mut t1 = f64::INFINITY;
            for (i, ((uvi, rvi), act)) in uv.iter().zip(rv.iter()).zip(iact.iter()).enumerate() {
                if act >= &meq && rvi > &0.0 {
                    let temp = uvi / rvi;
                    if temp < t1 {
                        t1 = temp;
                        idel = i;
                    }
                }
            }

            // Find the step length t2 to bring the slack variable to zero for the constraint we
            // are adding to the active set.
            // Store in ztn the rate at which the slack variable is increased. This is used to
            // update the objective value below.
            let (ztn, t2) = if dot(zv, zv).abs() <= f64::EPSILON {
                (0.0, f64::INFINITY)
            } else {
                let temp_ztn = dot(zv, aadd);
                (temp_ztn.abs(), slack / temp_ztn)
            };
            if t1 == f64::INFINITY && t2 == f64::INFINITY {
                return Err("optimization is infeasible");
            }

            // We will take a full step if t2 <= t1.
            let partial_step = t2 > t1;
            let step = if partial_step { t1 } else { t2 };

            // NOTE executing the next two steps only matters if t2 < inf, but the code is cleaner
            // without the check and doesn't run any slower

            // Update primal variable
            axpy(step, zv, &mut sol);
            // Update objective value
            obj += step * ztn * (step / 2.0 + u);

            // Update dual variable
            axpy(-step, &rv, &mut uv);
            u += step;

            partial_step
        } {
            // Remove constraint idel from the active set.
            let rlen = iact.len() * (iact.len() + 1) / 2;
            qr_delete(idel + 1, qmat, &mut rmat[..rlen]);
            // NOTE for some reason swap_remove doesn't perform better
            uv.remove(idel);
            iact.remove(idel);

            // NOTE we only need to execute this if t2 < inf but executing it doesn't actually take
            // any more time, and the code is simpler

            // We took a step in primal space, but only took a partial step.
            // So we need to update the slack variable that we are currently bringing to zero.
            slack = bvec[iadd] - dot(&sol, &aadd);
        }

        // Add constraint iadd to the active set.
        uv.push(u);
        iact.push(iadd);
        qr_insert(iact.len(), dv, qmat);
        let start = iact.len() * (iact.len() - 1) / 2;
        rmat[start..start + iact.len()].clone_from_slice(&dv[..iact.len()]);
    }

    // copy into lagrangian
    let mut lagr = vec![0.0; q];
    for (act, uvi) in iact.iter().copied().zip(uv) {
        lagr[act] = uvi;
    }

    Ok(Solution {
        obj: obj,
        sol: sol,
        lagr: lagr,
        iact: iact,
        iter: iter,
    })
}

#[cfg(test)]
mod tests {
    use super::{solve_qp, Solution};
    use approx::assert_relative_eq;

    fn assert_slices_eq(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert_relative_eq!(a, e, epsilon = 1e-6);
        }
    }

    fn verify_solution(
        res: &Solution,
        obj: f64,
        sol: Vec<f64>,
        lagr: Vec<f64>,
        mut iact: Vec<usize>,
        iter: usize,
    ) {
        assert_relative_eq!(res.obj, obj, epsilon = 1e-6);
        assert_slices_eq(&res.sol, &sol);
        assert_slices_eq(&res.lagr, &lagr);
        let mut res_iact = res.iact.to_owned();
        res_iact.sort_unstable();
        iact.sort_unstable();
        assert_eq!(res_iact, iact);
        assert_eq!(res.iter, iter);
    }

    #[test]
    fn it_works_on_one() {
        let mut q = [
            1., 0., 0., //
            0., 1., 0., //
            0., 0., 1., //
        ];
        let c = [0., -5., 0.];
        let a = [
            4., 3., 0., //
            -2., -1., 0., //
            0., 2., -1., //
        ];
        let b = [8., -2., 0.];
        let res = solve_qp(&mut q, &c, &a, &b, 0, false).unwrap();
        verify_solution(
            &res,
            -2.38095238095238,
            vec![0.476190476190476, 1.04761904761905, 2.0952380952381],
            vec![0., 0.238095238095238, 2.0952380952381],
            vec![1, 2],
            2,
        );
    }

    #[test]
    fn it_works_on_two() {
        let mut q = [
            0.0004523032,
            0.000509733,
            0.0005724848,
            0.0005049878,
            -0.00001126059,
            0.0001955939,
            0.0003306526,
            0.000509733,
            0.0006951339,
            0.0006417501,
            0.0006697231,
            -0.000001935067,
            0.0002421462,
            0.0003854708,
            0.0005724848,
            0.0006417501,
            0.0008401752,
            0.0006540224,
            -0.00001253942,
            0.0002540068,
            0.0004705115,
            0.0005049878,
            0.0006697231,
            0.0006540224,
            0.0008528426,
            0.00001678568,
            0.0002643017,
            0.0003590997,
            -0.00001126059,
            -0.000001935067,
            -0.00001253942,
            0.00001678568,
            0.0000602922,
            0.00002831196,
            -0.00002742019,
            0.0001955939,
            0.0002421462,
            0.0002540068,
            0.0002643017,
            0.00002831196,
            0.0001417604,
            0.0001441332,
            0.0003306526,
            0.0003854708,
            0.0004705115,
            0.0003590997,
            -0.00002742019,
            0.0001441332,
            0.002275453,
        ];
        let c = [
            -0.00006712571,
            -0.0000422238,
            -0.00007134523,
            -0.00002902528,
            -0.00002797279,
            -0.00004038785,
            -0.00004581844,
        ];
        let a = [
            1., 1., 1., 1., 1., 1., 1., //
            -1., 0., 0., 0., 0., 0., 0., //
            0., -1., 0., 0., 0., 0., 0., //
            0., 0., -1., 0., 0., 0., 0., //
            0., 0., 0., -1., 0., 0., 0., //
            0., 0., 0., 0., -1., 0., 0., //
            0., 0., 0., 0., 0., -1., 0., //
            0., 0., 0., 0., 0., 0., -1., //
            1., 0., 0., 0., 0., 0., 0., //
            0., 1., 0., 0., 0., 0., 0., //
            0., 0., 1., 0., 0., 0., 0., //
            0., 0., 0., 1., 0., 0., 0., //
            0., 0., 0., 0., 1., 0., 0., //
            0., 0., 0., 0., 0., 1., 0., //
            0., 0., 0., 0., 0., 0., 1., //
        ];
        let b = [
            1., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ];
        let res = solve_qp(&mut q, &c, &a, &b, 1, false).unwrap();
        verify_solution(
            &res,
            -2.88432729012647e-6,
            vec![
                0.0904705192225456,
                0.,
                0.,
                0.,
                0.5,
                0.40008532705569,
                0.00944415372176443,
            ],
            vec![
                4.95410837845179e-5,
                0.,
                5.3902979003092e-5,
                3.07049499047245e-5,
                8.46476095366474e-5,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                3.73182859218225e-5,
                0.,
                0.,
            ],
            vec![0, 2, 3, 4, 12],
            6,
        );
    }

    #[test]
    fn it_works_on_two_factorized() {
        let mut q = [
            47.020275545798654,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -102.58810490189579,
            91.02986883145324,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -120.75917430108004,
            2.6404344059002356,
            93.0572207588979,
            0.0,
            0.0,
            0.0,
            0.0,
            1.5558516312248016,
            -58.905941717492986,
            -10.778525176969659,
            70.2851188646282,
            0.0,
            0.0,
            0.0,
            19.310094620296482,
            -0.8579437739207997,
            -0.30003501316972114,
            -13.136491216427537,
            132.42308366971724,
            0.0,
            0.0,
            -38.15757340089673,
            -11.439622545483955,
            -6.707146720447394,
            -13.934623557372694,
            -84.27712229143113,
            168.80633800351532,
            0.0,
            1.8570104614383582,
            -5.010863752334357,
            -10.434912920178098,
            2.9218242844540985,
            9.962345883650197,
            -5.4978393185935435,
            22.380407666506038,
        ];
        let c = [
            -0.00006712571,
            -0.0000422238,
            -0.00007134523,
            -0.00002902528,
            -0.00002797279,
            -0.00004038785,
            -0.00004581844,
        ];
        let a = [
            1., 1., 1., 1., 1., 1., 1., //
            -1., 0., 0., 0., 0., 0., 0., //
            0., -1., 0., 0., 0., 0., 0., //
            0., 0., -1., 0., 0., 0., 0., //
            0., 0., 0., -1., 0., 0., 0., //
            0., 0., 0., 0., -1., 0., 0., //
            0., 0., 0., 0., 0., -1., 0., //
            0., 0., 0., 0., 0., 0., -1., //
            1., 0., 0., 0., 0., 0., 0., //
            0., 1., 0., 0., 0., 0., 0., //
            0., 0., 1., 0., 0., 0., 0., //
            0., 0., 0., 1., 0., 0., 0., //
            0., 0., 0., 0., 1., 0., 0., //
            0., 0., 0., 0., 0., 1., 0., //
            0., 0., 0., 0., 0., 0., 1., //
        ];
        let b = [
            1., 0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ];
        let res = solve_qp(&mut q, &c, &a, &b, 1, true).unwrap();
        verify_solution(
            &res,
            -2.88432729012647e-6,
            vec![
                0.0904705192225456,
                0.,
                0.,
                0.,
                0.5,
                0.40008532705569,
                0.00944415372176443,
            ],
            vec![
                4.95410837845179e-5,
                0.,
                5.3902979003092e-5,
                3.07049499047245e-5,
                8.46476095366474e-5,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
                3.73182859218225e-5,
                0.,
                0.,
            ],
            vec![0, 2, 3, 4, 12],
            6,
        );
    }

    #[test]
    fn it_works_on_three() {
        let mut q = [
            2., 0., 0., 0., 0., 0., //
            0., 2., 0., 0., 0., 0., //
            0., 0., 2., 0., 0., 0., //
            0., 0., 0., 2., 0., 0., //
            0., 0., 0., 0., 2., 0., //
            0., 0., 0., 0., 0., 2., //
        ];
        let c = [10., 10., 30., 20., 30., 20.];
        let a = [
            -1., 0., 0., 0., 0., 0., //
            1., 0., 0., 0., 0., 0., //
            0., -1., -1., -1., 0., 0., //
            0., 1., 1., 1., 0., 0., //
            0., 0., 0., 0., -1., -1., //
            0., 0., 0., 0., 1., 1., //
            1., 1., 0., 0., 0., 0., //
            0., 0., 1., 0., 1., 0., //
            0., 0., 0., 1., 0., 1., //
            -1., 0., 0., 0., 0., 0., //
            0., -1., 0., 0., 0., 0., //
            0., 0., -1., 0., 0., 0., //
            0., 0., 0., -1., 0., 0., //
            0., 0., 0., 0., -1., 0., //
            0., 0., 0., 0., 0., -1., //
        ];
        let b = [
            -3.999, 4.001, -4.999, 5.001, -0.999, 1.001, 7.001, 1.001, 2.001, 0.001, 0.001, 0.001,
            0.001, 0.001, 0.001,
        ];
        let res = solve_qp(&mut q, &c, &a, &b, 0, false).unwrap();
        verify_solution(
            &res,
            167.630019,
            vec![3.999, 3.002, 0.747, 1.25, 0.248, 0.751],
            vec![
                33.488, 0., 31.494, 0., 30.496, 0., 15.49, 0., 8.994, 0., 0., 0., 0., 0., 0.,
            ],
            vec![0, 2, 4, 6, 8],
            7,
        );
    }

    #[test]
    fn it_works_on_four_unconstrained() {
        let mut q = [
            13., 18., -6., //
            18., 27., -9., //
            -6., -9., 4., //
        ];
        let c = [4., 0., 100.];
        let res = solve_qp(&mut q, &c, &[], &[], 0, false).unwrap();
        verify_solution(
            &res,
            -5008.,
            vec![-4., -30.6666666666666, -100.],
            vec![],
            vec![],
            0,
        );
    }

    #[test]
    fn it_works_on_four() {
        let mut q = [
            13., 18., -6., //
            18., 27., -9., //
            -6., -9., 4., //
        ];
        let c = [4., 0., 100.];
        let a = [0., 0., 1.];
        let b = [25.];
        let res = solve_qp(&mut q, &c, &a, &b, 1, false).unwrap();
        verify_solution(&res, 2804.5, vec![-4., 11., 25.], vec![125.], vec![0], 1);
    }

    #[test]
    fn it_works_on_five() {
        let mut q = [
            2., 0., 0., 0., 0., 0., //
            0., 2., 0., 0., 0., 0., //
            0., 0., 2., 0., 0., 0., //
            0., 0., 0., 2., 0., 0., //
            0., 0., 0., 0., 2., 0., //
            0., 0., 0., 0., 0., 2., //
        ];
        let c = [10., 10., 30., 20., 30., 20.];
        let a = [
            1., 0., 0., 0., 0., 0., //
            0., 1., 1., 1., 0., 0., //
            0., 0., 0., 0., 1., 1., //
            1., 1., 0., 0., 0., 0., //
            0., 0., 1., 0., 1., 0., //
            0., 0., 0., 1., 0., 1., //
            -1., 0., 0., 0., 0., 0., //
            0., -1., 0., 0., 0., 0., //
            0., 0., -1., 0., 0., 0., //
            0., 0., 0., -1., 0., 0., //
            0., 0., 0., 0., -1., 0., //
            0., 0., 0., 0., 0., -1., //
        ];
        let b = [4., 5., 1., 7.001, 1.001, 2.001, 0., 0., 0., 0., 0., 0.];
        let res = solve_qp(&mut q, &c, &a, &b, 3, false).unwrap();
        verify_solution(
            &res,
            167.72550375,
            vec![4., 3.001, 0.74875, 1.25025, 0.24925, 0.75075],
            vec![
                33.4955, 31.4975, 30.4985, 15.4955, 0., 8.997, 0., 0., 0., 0., 0., 0.,
            ],
            vec![0, 1, 2, 3, 5],
            7,
        );
    }

    #[test]
    fn it_works_on_six() {
        let mut q = [
            1., 0., 0., 0., 0., //
            0., 1., 0., 0., 0., //
            0., 0., 1., 0., 0., //
            0., 0., 0., 1., 0., //
            0., 0., 0., 0., 1., //
        ];
        let c = [5., 0.5, 0., -0.2, -2.];
        let a = [
            -1., 0., 0., 0., 0., //
            0., -1., 0., 0., 0., //
            0., 0., -1., 0., 0., //
            0., 0., 0., -1., 0., //
            0., 0., 0., 0., -1., //
            1., 0., 0., 0., 0., //
            0., 1., 0., 0., 0., //
            0., 0., 1., 0., 0., //
            0., 0., 0., 1., 0., //
            0., 0., 0., 0., 1., //
        ];
        let b = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.];
        let res = solve_qp(&mut q, &c, &a, &b, 0, false).unwrap();
        verify_solution(
            &res,
            -6.145,
            vec![-1., -0.5, 0., 0.2, 1.],
            vec![4., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            vec![0, 9],
            2,
        );
    }

    #[test]
    fn it_works_on_seven() {
        let mut q = [
            1., 0., 0., 0., 0., //
            0., 1., 0., 0., 0., //
            0., 0., 1., 0., 0., //
            0., 0., 0., 1., 0., //
            0., 0., 0., 0., 1., //
        ];
        let c = [
            -0.73727161,
            -0.75526241,
            -0.04741426,
            0.11260887,
            0.11260887,
        ];
        let a = [
            3.6, -3.4, -3.8, 1.6, 1.6, //
            0., -1.9, -1.7, -4., -4., //
            -9.72, -8.67, 0., 0., 0., //
        ];
        let b = [1.02, 0.03, 0.081];
        let res = solve_qp(&mut q, &c, &a, &b, 3, false).unwrap();
        verify_solution(
            &res,
            0.0393038880729888,
            vec![0.07313507, -0.09133482, -0.08677699, 0.03638213, 0.03638213],
            vec![0.0440876, 0.01961271, 0.08465554],
            vec![0, 1, 2],
            3,
        );
    }

    #[test]
    fn it_works_on_generated() {
        let m = 10_usize;
        let n = 2 * m; // enforce even
        let n2 = n * n;
        let mut q = vec![0.0; n2];
        for i in (0..n2).step_by(n + 1) {
            q[i] = 1.0;
        }
        let c = vec![0.0; n];
        let mut a = vec![0.0; n2 + n];
        for i in (0..n2).step_by(n + 1) {
            a[i] = -1.0;
        }
        for i in (0..n).step_by(2) {
            a[n2 + i] = -1.0 + i as f64 * 1e-6;
        }
        for i in (1..n).step_by(2) {
            a[n2 + i] = 1.0 + i as f64 * 1e-6;
        }
        let mut b = vec![-1.0; n + 1];
        b[n] = -1.01;

        let res = solve_qp(&mut q, &c, &a, &b, 0, false).unwrap();
        assert!(res.obj > m as f64);
        assert_eq!(res.iact.len(), m + 1);
        assert_eq!(res.iter, n + 1);
    }

    #[test]
    fn it_works_for_barbiggs() {
        let mut q = [
            0.00005, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0.00005, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0.000075, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0.00005, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0.00005, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0.000075, 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0.00005, 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0.00005, 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0.000075, 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0.00005, 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.00005, 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.000075, 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.00005, 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.00005, 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.000075, //
        ];
        let c = [
            2.3, 1.7, 2.2, 2.3, 1.7, 2.2, 2.3, 1.7, 2.2, 2.3, 1.7, 2.2, 2.3, 1.7, 2.2,
        ];
        let a = [
            1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            -1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., -1., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., -1., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., -1., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., -1., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 1., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 1., //
            -1., -1., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., -1., -1., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., -1., -1., -1., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., -1., -1., -1., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., -1., -1., //
            -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -1., //
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., //
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., //
        ];
        let b = [
            7., 7., 7., 6., 7., 6., 7., 7., 7., 6., 7., 6., 7., 7., 7., 6., 7., 6., 7., 7., 7., 6.,
            7., 6., -60., -50., -70., -85., -100., -8., -43., -3., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 21., 57., 16., 90., 120., 60., 90., 120., 60., 90., 120., 60., 90.,
            120., 60.,
        ];
        let res = solve_qp(&mut q, &c, &a, &b, 0, false).unwrap();
        verify_solution(
            &res,
            663.2301125,
            vec![
                8., 49., 3., 1., 56., 0., 1., 63., 6., 3., 70., 12., 5., 77., 18.,
            ],
            vec![
                2.30005, 0., 0., 0., 0.08715, 0., 0., 0., 0., 0., 1.78995, 0.29775, 0., 0., 0., 0.,
                1.19305, 0.19815, 0., 0., 0., 0., 0.5964, 0.0989, 1.6153, 0., 2.30005, 2.30015,
                2.30025, 2.98515, 0., 0.584925, 0., 0., 1.90225, 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
            vec![0, 4, 10, 11, 16, 17, 22, 23, 24, 26, 27, 28, 29, 31, 34],
            29,
        );
    }

    #[test]
    fn it_errors_for_invalid_q_size() {
        let msg = solve_qp(&mut [1., 0., 0.], &[0., 5.], &[], &[], 0, false).unwrap_err();
        assert_eq!(msg, "qmat was not appropriate size given cvec");
    }

    #[test]
    fn it_errors_for_invalid_c_size() {
        let msg = solve_qp(&mut [1., 0., 0., 1.], &[0., 5.], &[0.], &[], 0, false).unwrap_err();
        assert_eq!(msg, "amat was not appropriate size given cvec and bvec");
    }

    #[test]
    fn it_errors_for_non_positive_definite_q() {
        let msg = solve_qp(&mut [-1.], &[0.], &[], &[], 0, false).unwrap_err();
        assert_eq!(msg, "matrix not positive definite");
    }

    #[test]
    fn it_errors_for_invalid_constraints() {
        let msg = solve_qp(&mut [1.], &[0.], &[-1., 1.], &[-1., -1.], 0, false).unwrap_err();
        assert_eq!(msg, "optimization is infeasible");
    }
}
