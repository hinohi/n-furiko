use std::ops::Mul;

use generic_array::ArrayLength;
use nalgebra::U3;
use nalgebra::{Dim, DimName, Scalar, VectorN};
use num_traits::Zero;
use typenum::{bit::B1, UInt, UTerm};

#[derive(Debug)]
struct NFuriko<N, D>
where
    N: Scalar,
    D: Dim + DimName,
    D::Value: Mul<UInt<UTerm, B1>>,
    <D::Value as Mul<UInt<UTerm, B1>>>::Output: ArrayLength<N>,
{
    length: VectorN<N, D>,
    mass: VectorN<N, D>,
}

impl<N, D> NFuriko<N, D>
where
    N: Scalar,
    D: Dim + DimName,
    D::Value: Mul<UInt<UTerm, B1>>,
    <D::Value as std::ops::Mul<UInt<UTerm, B1>>>::Output: ArrayLength<N>,
{
    fn new_all(length: N, mass: N) -> NFuriko<N, D> {
        NFuriko {
            length: VectorN::<N, D>::from_element(length),
            mass: VectorN::<N, D>::from_element(mass),
        }
    }
}

#[derive(Debug)]
struct PhaseSpaceEuler<N, D>
where
    N: Scalar,
    D: Dim + DimName,
    D::Value: Mul<UInt<UTerm, B1>>,
    <D::Value as Mul<UInt<UTerm, B1>>>::Output: ArrayLength<N>,
{
    position: VectorN<N, D>,
    velocity: VectorN<N, D>,
}

impl<N, D> PhaseSpaceEuler<N, D>
where
    N: Scalar + Zero,
    D: Dim + DimName,
    D::Value: Mul<UInt<UTerm, B1>>,
    <D::Value as std::ops::Mul<UInt<UTerm, B1>>>::Output: ArrayLength<N>,
{
    fn new_all(angle: N) -> PhaseSpaceEuler<N, D> {
        PhaseSpaceEuler {
            position: VectorN::<N, D>::from_element(angle),
            velocity: VectorN::<N, D>::zeros(),
        }
    }
}

fn main() {
    let furiko = NFuriko::<f64, U3>::new_all(1.0, 1.0);
    let ps = PhaseSpaceEuler::<f64, U3>::new_all(0.5);
    println!("{:?}", furiko);
    println!("{:?}", ps);
}
