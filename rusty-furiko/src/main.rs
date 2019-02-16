use nalgebra::{DMatrix, DVector};

const G: f64 = 9.8;

#[derive(Debug)]
struct NFuriko {
    n: usize,
    length: DVector<f64>,
    mass: DVector<f64>,
    accumulated_mass: DVector<f64>,
}

impl NFuriko {
    fn new_all(n: usize, length: f64, mass: f64) -> NFuriko {
        NFuriko {
            n,
            length: DVector::from_element(n, length),
            mass: DVector::from_element(n, mass),
            accumulated_mass: DVector::from_iterator(n, (0..n).map(|i| (n - i) as f64 * mass)),
        }
    }

    fn position_energy(&self, position: &[f64]) -> f64 {
        let mut u = 0.0;
        for i in 0..self.n {
            u -= self.accumulated_mass[i] * self.length[i] * position[i].cos();
        }
        u * G
    }

    fn physical_energy(&self, position: &[f64], velocity: &[f64]) -> f64 {
        let mut t = 0.0;
        for i in 0..self.n {
            t += self.accumulated_mass[i]
                * self.length[i]
                * self.length[i]
                * velocity[i]
                * velocity[i];
            for j in 0..i {
                t += self.accumulated_mass[i]
                    * self.length[i]
                    * self.length[j]
                    * velocity[i]
                    * velocity[j]
                    * (position[i] - position[j]).cos();
            }
            for j in i + 1..self.n {
                t += self.accumulated_mass[j]
                    * self.length[i]
                    * self.length[j]
                    * velocity[i]
                    * velocity[j]
                    * (position[i] - position[j]).cos();
            }
        }
        t * 0.5
    }

    fn calc_acceleration(&self, position: &[f64], velocity: &[f64]) -> DVector<f64> {
        let mut b = DVector::from_iterator(self.n, velocity.iter().map(|v| v * v));
        let c = DMatrix::from_fn(self.n, self.n, |x, y| {
            self.length[x]
                * self.length[y]
                * (if x > y {
                    self.accumulated_mass[x] * (position[y] - position[x]).sin()
                } else if x < y {
                    self.accumulated_mass[y] * (position[y] - position[x]).sin()
                } else {
                    0.0
                })
        });
        b = c * b;
        b.iter_mut()
            .zip(self.accumulated_mass.iter())
            .zip(self.length.iter())
            .zip(position.iter())
            .map(|(((v, m), l), p)| *v -= G * m * l * p.sin())
            .last();
        let a = DMatrix::from_fn(self.n, self.n, |i, j| {
            self.accumulated_mass[if i > j { i } else { j }]
                * self.length[i]
                * self.length[j]
                * (position[i] - position[j]).cos()
        });
        let d = a.qr();
        if !d.solve_mut(&mut b) {
            eprintln!("Fail");
        }
        b
    }
}

#[derive(Debug)]
struct PhaseSpace2DEuler<'a> {
    furiko: &'a NFuriko,
    position: DVector<f64>,
    velocity: DVector<f64>,
}

impl<'a> PhaseSpace2DEuler<'a> {
    fn new_all(furiko: &'a NFuriko, angle: f64) -> PhaseSpace2DEuler<'a> {
        PhaseSpace2DEuler {
            furiko,
            position: DVector::from_element(furiko.n, angle),
            velocity: DVector::zeros(furiko.n),
        }
    }

    fn iterate(&mut self, dt: f64) {
        let a = self
            .furiko
            .calc_acceleration(&self.position.as_slice(), &self.velocity.as_slice());
        self.position += &self.velocity * dt;
        self.velocity += a * dt;
    }

    fn total_energy(&self) -> f64 {
        let p = self.position.as_slice();
        let v = self.velocity.as_slice();
        self.furiko.position_energy(&p) + self.furiko.physical_energy(&p, &v)
    }
}

fn main() {
    let furiko = NFuriko::new_all(3, 0.5, 1.0);
    let mut ps = PhaseSpace2DEuler::new_all(&furiko, 3.0);
    let mut t = 0.0;
    let dt = 2_f64.powi(-15);
    let out_interval = 2.0_f64.powi(-7);
    while t < 10.0 {
        if t % out_interval == 0.0 {
            println!("{} {}", t, ps.total_energy());
        }
        ps.iterate(dt);
        t += dt;
    }
}
