use nalgebra::{DMatrix, DVector};

const G: f64 = 9.8;


pub trait FurikoSpec {

}

#[derive(Debug)]
pub struct FurikoSpec2D {
    n: usize,
    length: Vec<f64>,
    mass: Vec<f64>,
    accumulated_1: Vec<f64>,
    accumulated_2: Vec<(f64, usize, usize)>,
}

fn accumulate(length: &[f64], mass: &[f64]) -> (Vec<f64>, Vec<(f64, usize, usize)>) {
    let n = length.len();
    let mut am = Vec::from(mass.clone());
    for i in (0..n - 1).rev() {
        am[i] += am[i + 1];
    }
    let a1 = length
        .iter()
        .zip(am.iter())
        .map(|(l, m)| l * m)
        .collect::<Vec<_>>();
    let mut a2 = Vec::with_capacity(n * n);
    for x in 0..n {
        for y in 0..n {
            a2.push((am[if x > y { x } else { y }] * length[x] * length[y], x, y));
        }
    }
    (a1, a2)
}

impl FurikoSpec2D {
    fn new_all(n: usize, length: f64, mass: f64) -> FurikoSpec2D {
        let length = vec![length; n];
        let mass = vec![mass; n];
        let (a1, a2) = accumulate(&length, &mass);
        FurikoSpec2D {
            n,
            length,
            mass,
            accumulated_1: a1,
            accumulated_2: a2,
        }
    }

    fn position_energy(&self, position: &[f64]) -> f64 {
        let mut u = 0.0;
        for (a, p) in self.accumulated_1.iter().zip(position.iter()) {
            u -= a * p.cos();
        }
        u * G
    }

    fn physical_energy(&self, position: &[f64], velocity: &[f64]) -> f64 {
        let mut t = 0.0;
        for (a, i, j) in self.accumulated_2.iter() {
            let i = *i;
            let j = *j;
            t += a * velocity[i] * velocity[j] * (position[i] - position[j]).cos();
        }
        t * 0.5
    }

    fn calc_acceleration(&self, position: &[f64], velocity: &[f64]) -> DVector<f64> {
        let mut b = DVector::from_iterator(self.n, velocity.iter().map(|v| v * v));
        let c = DMatrix::from_iterator(
            self.n,
            self.n,
            self.accumulated_2
                .iter()
                .map(|(a, i, j)| a * (position[*i] - position[*j]).sin()),
        );
        b = c * b;
        b.iter_mut()
            .zip(self.accumulated_1.iter())
            .zip(position.iter())
            .map(|((v, a), p)| *v -= G * a * p.sin())
            .last();
        let a = DMatrix::from_iterator(
            self.n,
            self.n,
            self.accumulated_2
                .iter()
                .map(|(a, i, j)| a * (position[*i] - position[*j]).cos()),
        );
        let d = a.qr();
        if !d.solve_mut(&mut b) {
            eprintln!("Fail");
        }
        b
    }
}
