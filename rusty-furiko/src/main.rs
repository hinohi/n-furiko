use nalgebra::DVector;

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
}

#[derive(Debug)]
struct PhaseSpace2DEuler {
    n: usize,
    position: DVector<f64>,
    velocity: DVector<f64>,
}

impl PhaseSpace2DEuler {
    fn new_all(n: usize, angle: f64) -> PhaseSpace2DEuler {
        PhaseSpace2DEuler {
            n,
            position: DVector::from_element(n, angle),
            velocity: DVector::zeros(n),
        }
    }
}

fn main() {
    let furiko = NFuriko::new_all(3, 1.0, 1.0);
    let ps = PhaseSpace2DEuler::new_all(3, 0.5);
    println!("{:?}", furiko);
    println!("{:?}", ps);
}
