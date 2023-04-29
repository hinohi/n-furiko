

pub trait Energy {

    fn position_energy(&self, position: &[f64]) -> f64;

    fn physical_energy(&self, position: &[f64], velocity: &[f64]);
}
