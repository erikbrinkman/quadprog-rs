use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quadprog::solve_qp;

fn barbiggs_benchmark(crit: &mut Criterion) {
    let q = [
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
        7., 7., 7., 6., 7., 6., 7., 7., 7., 6., 7., 6., 7., 7., 7., 6., 7., 6., 7., 7., 7., 6., 7.,
        6., -60., -50., -70., -85., -100., -8., -43., -3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 21., 57., 16., 90., 120., 60., 90., 120., 60., 90., 120., 60., 90., 120., 60.,
    ];
    crit.bench_function("barbiggs", |bench| {
        bench.iter(|| {
            let mut qm = q.clone();
            solve_qp(
                black_box(&mut qm),
                black_box(&c),
                black_box(&a),
                black_box(&b),
                black_box(0),
                black_box(false),
            )
            .unwrap();
        })
    });
}

fn large_benchmark(crit: &mut Criterion) {
    for (name, size) in [("medium", 100_usize)] {
        let n = 2 * size; // enforce even
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
        crit.bench_function(name, |bench| {
            bench.iter(|| {
                let mut qm = q.clone();
                let res = solve_qp(
                    black_box(&mut qm),
                    black_box(&c),
                    black_box(&a),
                    black_box(&b),
                    black_box(0),
                    black_box(false),
                )
                .unwrap();
                assert_eq!(res.iter - res.iact.len(), size);
            })
        });
    }
}

criterion_group!(benches, barbiggs_benchmark, large_benchmark);
criterion_main!(benches);