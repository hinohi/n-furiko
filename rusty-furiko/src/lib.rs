use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;

use nalgebra::{DMatrix, DVector};

mod furiko;

pub struct Furiko {
    spec: Spec,
    position: DVector<f64>,
    velocity: DVector<f64>,
    t: f64,
    dt: i32,
}

impl Furiko {
    fn evaluate_rk44(&mut self, dt: f64) {
        let a1 = self
            .spec
            .calc_acceleration(&self.position.as_slice(), &self.velocity.as_slice());

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
        let a2 = self.spec.calc_acceleration(&x1.as_slice(), &v1.as_slice());

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
        let a3 = self.spec.calc_acceleration(&x2.as_slice(), &v2.as_slice());

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
        let a4 = self.spec.calc_acceleration(&x3.as_slice(), &v3.as_slice());

        self.position += &self.velocity * (dt / 6.0);
        self.position += v1 * (dt / 3.0);
        self.position += v2 * (dt / 3.0);
        self.position += v3 * (dt / 6.0);
        self.velocity += a1 * (dt / 6.0);
        self.velocity += a2 * (dt / 3.0);
        self.velocity += a3 * (dt / 3.0);
        self.velocity += a4 * (dt / 6.0);
    }

    fn mod_pi(&mut self) {
        self.position.iter_mut().map(|p| *p %= 2.0 * PI).last();
    }
}

impl Furiko {
    pub fn new(n: usize) -> Furiko {
        Furiko {
            spec: Spec::new_all(n, 0.5, 1.0),
            position: DVector::from_element(n, 3.0),
            velocity: DVector::zeros(n),
            t: 0.0,
            dt: -20,
        }
    }

    pub fn get_energy(&self) -> f64 {
        let p = self.position.as_slice();
        let v = self.velocity.as_slice();
        self.spec.position_energy(&p) + self.spec.physical_energy(&p, &v)
    }

    pub fn set_dt(&mut self, n: i32) {
        self.dt = n;
    }

    pub fn get_time(&self) -> f64 {
        self.t
    }

    pub fn evaluate(&mut self, resolution: i32) {
        assert!(self.dt <= resolution);
        let dt = 2_f64.powi(self.dt);
        for _ in 0..1 << resolution - self.dt {
            self.evaluate_rk44(dt);
        }
        self.t += 2_f64.powi(resolution);
        self.mod_pi();
    }

    pub fn get_x(&self) -> Vec<(f64, f64)> {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut ret = Vec::with_capacity(self.position.len());
        for (p, l) in self.position.as_slice().iter().zip(self.spec.length.iter()) {
            x += l * p.sin();
            y -= l * p.cos();
            ret.push((x, y));
        }
        ret
    }
}

#[cfg(target_arc = "wasm32")]
mod js {
    use std::cell::RefCell;
    use std::rc::Rc;

    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;

    use crate::Furiko;

    fn window() -> web_sys::Window {
        web_sys::window().expect("no global `window` exists")
    }

    fn request_animation_frame(f: &Closure<FnMut()>) {
        window()
            .request_animation_frame(f.as_ref().unchecked_ref())
            .expect("should register `requestAnimationFrame` OK");
    }

    fn document() -> web_sys::Document {
        window()
            .document()
            .expect("should have a document on window")
    }

    fn body() -> web_sys::HtmlElement {
        document().body().expect("document should have a body")
    }

    #[wasm_bindgen(start)]
    pub fn start() -> Result<(), JsValue> {
        let width = 600.0;
        let height = 600.0;
        let scale = 100.0;
        let canvas = document()
            .create_element("canvas")?
            .dyn_into::<web_sys::HtmlCanvasElement>()?;
        body().append_child(&canvas)?;
        canvas.set_width(width as u32);
        canvas.set_height(height as u32);
        canvas.style().set_property("border", "solid")?;
        let context = canvas
            .get_context("2d")?
            .unwrap()
            .dyn_into::<web_sys::CanvasRenderingContext2d>()?;
        context.set_fill_style(&JsValue::from_str("rgba(1, 1, 1, 1)"));

        let mut furiko = Furiko::new(6);
        furiko.set_dt(-12);

        let f = Rc::new(RefCell::new(None));
        let g = f.clone();
        *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            furiko.evaluate(-6);

            context.rect(0.0, 0.0, width, height);
            context.fill();

            context.begin_path();
            context.move_to(width * 0.5, height * 0.5);
            for (x, y) in furiko.get_x() {
                let x = width * 0.5 + x * scale;
                let y = height * 0.5 - y * scale;
                context.line_to(x, y);
            }
            context.stroke();

            request_animation_frame(f.borrow().as_ref().unwrap());
        }) as Box<FnMut()>));

        request_animation_frame(g.borrow().as_ref().unwrap());
        Ok(())
    }
}