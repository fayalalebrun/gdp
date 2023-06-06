use std::ops::Mul;

use nalgebra::Point3;
use nalgebra_glm::{Mat4, Vec3};

pub struct Camera {
    pub eye: Point3<f32>,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            eye: Point3::new(1.0, 1.0, 0.),
        }
    }

    pub fn rotate_y(&mut self, angle: f32) {
        self.eye = Mat4::new_rotation(Vec3::y_axis().mul(angle)).transform_point(&self.eye);
    }

    pub fn rotate_h(&mut self, angle: f32) {
        let axis = (self.eye - Point3::origin()).cross(&Vec3::y());

        self.eye = Mat4::new_rotation(axis.mul(angle)).transform_point(&self.eye);
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_lh(&self.eye, &Point3::new(0., 0., 0.), &Vec3::new(0., 1., 0.))
    }

    pub fn change_zoom(&mut self, modifier: f32) {
        self.eye *= modifier;
    }

    pub fn perspective(&self, aspect_ratio: f32) -> Mat4 {
        nalgebra_glm::perspective_fov_lh_no(
            60.0_f32.to_radians(),
            aspect_ratio,
            1.0,
            0.0001,
            1000.0,
        )
    }
}
