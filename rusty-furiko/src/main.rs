use nalgebra::{DMatrix, DVector};

const G: f64 = 9.8;

#[derive(Debug)]
struct Spec {
    n: usize,
    length: Vec<f64>,
    mass: Vec<f64>,
    accumulated_1: Vec<f64>,
    accumulated_2: Vec<(f64, usize, usize)>,
}

impl Spec {
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
    fn new_all(n: usize, length: f64, mass: f64) -> Spec {
        let length = vec![length; n];
        let mass = vec![mass; n];
        let (a1, a2) = Spec::accumulate(&length, &mass);
        Spec {
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

#[derive(Debug)]
struct PhaseSpace {
    position: DVector<f64>,
    velocity: DVector<f64>,
}

impl PhaseSpace {
    fn new_all(n: usize, angle: f64) -> PhaseSpace {
        PhaseSpace {
            position: DVector::from_element(n, angle),
            velocity: DVector::zeros(n),
        }
    }

    fn total_energy(&self, spec: &Spec) -> f64 {
        let p = self.position.as_slice();
        let v = self.velocity.as_slice();
        spec.position_energy(&p) + spec.physical_energy(&p, &v)
    }

    fn evaluate_rk44(&mut self, spec: &Spec, dt: f64) {
        let a1 = spec.calc_acceleration(&self.position.as_slice(), &self.velocity.as_slice());

        let mut x1 = self.position.clone();
        let mut v1 = self.velocity.clone();
        x1.iter_mut()
            .zip(self.velocity.iter())
            .map(|(x, v)| *x += v * dt * 0.5)
            .last();
        v1.iter_mut()
            .zip(a1.iter())
            .map(|(v, a)| *v += a * dt * 0.5)
            .last();
        let a2 = spec.calc_acceleration(&x1.as_slice(), &v1.as_slice());

        let mut x2 = self.position.clone();
        let mut v2 = self.velocity.clone();
        x2.iter_mut()
            .zip(v1.iter())
            .map(|(x, v)| *x += v * dt * 0.5)
            .last();
        v2.iter_mut()
            .zip(a2.iter())
            .map(|(v, a)| *v += a * dt * 0.5)
            .last();
        let a3 = spec.calc_acceleration(&x2.as_slice(), &v2.as_slice());

        let mut x3 = self.position.clone();
        let mut v3 = self.velocity.clone();
        x3.iter_mut()
            .zip(v2.iter())
            .map(|(x, v)| *x += v * dt * 0.5)
            .last();
        v3.iter_mut()
            .zip(a3.iter())
            .map(|(v, a)| *v += a * dt * 0.5)
            .last();
        let a4 = spec.calc_acceleration(&x3.as_slice(), &v3.as_slice());

        self.position += self.velocity.clone() * (dt / 6.0);
        self.position += v1 * (dt / 3.0);
        self.position += v2 * (dt / 3.0);
        self.position += v3 * (dt / 6.0);
        self.velocity += a1 * (dt / 6.0);
        self.velocity += a2 * (dt / 3.0);
        self.velocity += a3 * (dt / 3.0);
        self.velocity += a4 * (dt / 6.0);
    }
}

struct Furiko {
    spec: Spec,
    phase_space: PhaseSpace,
    t: f64,
    dt: i32,
}

impl Furiko {
    fn new(n: usize) -> Furiko {
        Furiko {
            spec: Spec::new_all(n, 0.5, 1.0),
            phase_space: PhaseSpace::new_all(n, 3.0),
            t: 0.0,
            dt: -20,
        }
    }

    fn set_dt(&mut self, n: i32) {
        self.dt = n;
    }

    fn energy(&self) -> f64 {
        self.phase_space.total_energy(&self.spec)
    }

    fn evaluate(&mut self, resolution: i32) {
        assert!(self.dt <= resolution);
        let dt = 2_f64.powi(self.dt);
        for _ in 0..1 << resolution - self.dt {
            self.phase_space.evaluate_rk44(&self.spec, dt);
        }
        self.t += 2_f64.powi(resolution);
    }
}

fn main() {
    let mut furiko = Furiko::new(3);
    furiko.set_dt(-20);
    let resolution = -7;
    println!("{} {}", furiko.t, furiko.energy());
    for _ in 0..10 << (-resolution) {
        furiko.evaluate(resolution);
        println!("{} {}", furiko.t, furiko.energy());
    }
}
