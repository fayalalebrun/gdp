use std::collections::HashMap;

use egui::{Color32, Pos2, Rect, Rounding};
use glium::{
    backend::Facade,
    glutin::event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    Frame, Surface,
};

use nalgebra::{Matrix3, Vector3};
use nalgebra_glm::{Mat4, Vec2};

use crate::{Matrices, Model};

use self::{camera::Camera, mesh::Mesh};

mod camera;
mod gui;
mod mesh;

pub struct State {
    model: Model,
    mesh: Mesh,
    camera: Camera,
    right: DeltaMouse,
    left: SelectionMouse,
    resolution: Vec2,
    transformation: Matrix3<f32>,
    result: Result<(), String>,
}

struct SelectionMouse {
    pressed: bool,
    start_position: Vec2,
    curr_position: Vec2,
    selection_rect: Option<(Vec2, Vec2)>,
}

impl SelectionMouse {
    pub fn handle_position(&mut self, position: Vec2) {
        if self.pressed {
            self.curr_position = position
        } else {
            self.start_position = position;
            self.curr_position = position;
        }
    }

    pub fn handle_pressed(&mut self, pressed: bool) {
        if self.pressed && !pressed {
            self.selection_rect = Some(self.current_rect());
        }
        self.pressed = pressed;
    }

    pub fn current_rect(&self) -> (Vec2, Vec2) {
        (
            self.curr_position.inf(&self.start_position),
            self.curr_position.sup(&self.start_position),
        )
    }

    pub fn current_rect_normalized(&self, denominator: &Vec2) -> (Vec2, Vec2) {
        self.normalize_area(self.current_rect(), denominator)
    }

    pub fn normalize_area(&self, numerator: (Vec2, Vec2), denominator: &Vec2) -> (Vec2, Vec2) {
        let (first, second) = numerator;
        (
            first.component_div(denominator),
            second.component_div(denominator),
        )
    }

    pub fn selection_rect(&mut self, denominator: &Vec2) -> Option<(Vec2, Vec2)> {
        let rect = self
            .selection_rect
            .map(|s| self.normalize_area(s, denominator))
            .take();
        self.selection_rect = None;
        rect
    }
}

impl Default for SelectionMouse {
    fn default() -> Self {
        Self {
            pressed: false,
            start_position: Default::default(),
            curr_position: Default::default(),
            selection_rect: None,
        }
    }
}

struct DeltaMouse {
    pressed: bool,
    last_position: Vec2,
    new_position: Vec2,
    last_delta: Vec2,
}

impl DeltaMouse {
    pub fn normalized_delta(&mut self, resolution: Vec2) -> Vec2 {
        self.last_delta = self.new_position - self.last_position;
        self.last_position = self.new_position;
        if self.pressed {
            self.last_delta.component_div(&resolution)
        } else {
            Vec2::new(0., 0.)
        }
    }
}

impl Default for DeltaMouse {
    fn default() -> Self {
        Self {
            pressed: false,
            last_position: Default::default(),
            new_position: Default::default(),
            last_delta: Default::default(),
        }
    }
}

impl State {
    pub fn paint_gui(&mut self, ctx: &egui::Context) {
        egui::Window::new("Tools").show(ctx, |ui| {
            ui.label("Transformation:");
            egui::Grid::new("matrix").num_columns(3).show(ui, |ui| {
                for i in 0..3 {
                    for j in 0..3 {
                        ui.add(egui::DragValue::new(&mut self.transformation[(i, j)]).speed(0.01));
                    }
                    ui.end_row();
                }

                if ui.button("Apply").clicked() {
                    self.result = self.transform();
                }
            });

            if let Err(e) = self.result.as_ref() {
                ui.colored_label(Color32::RED, e);
            }
        });

        if self.left.pressed {
            let egui_extent = ctx.debug_painter().clip_rect();
            let egui_extent = Vec2::new(egui_extent.width(), egui_extent.height());
            let (first, second) = self
                .left
                .current_rect_normalized(&self.resolution.component_div(&egui_extent));

            ctx.debug_painter().rect_filled(
                Rect::from_two_pos(Pos2::new(first.x, first.y), Pos2::new(second.x, second.y)),
                Rounding::none(),
                Color32::from_rgba_unmultiplied(255, 255, 255, 128),
            );
        }
    }

    pub fn draw(&mut self, frame: &mut Frame, facade: &dyn Facade) {
        let (width, height) = frame.get_dimensions();
        let resolution = Vec2::new(width as f32, height as f32);
        self.resolution = resolution;

        let camera_delta = self.right.normalized_delta(resolution);

        self.camera.rotate_y(camera_delta.x * 1.);
        self.camera.rotate_h(camera_delta.y * 1.);

        let model = Mat4::identity();
        let view = self.camera.view();
        let perspective = self.camera.perspective(resolution.x / resolution.y);

        if let Some((bottom_left, top_right)) =
            self.left.selection_rect(&resolution).map(|(v1, v2)| {
                // Need to negate y because in OpenGL NDC coordinates +y is at the bottom.
                let x_neg = Vec2::new(1.0, -1.0);
                (
                    ((v1.add_scalar(-0.5)) * 2.0).component_mul(&x_neg),
                    ((v2.add_scalar(-0.5)) * 2.0).component_mul(&x_neg),
                )
            })
        {
            let contained_tris =
                self.model
                    .triangles_in_rect(perspective * view * model, bottom_left, top_right);

            self.mesh.set_selected(facade, &self.model, &contained_tris);
        }

        self.mesh
            .draw(frame, Mat4::identity(), view, perspective, self.camera.eye);
    }

    pub fn handle_window_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(VirtualKeyCode::A),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.camera.rotate_y(0.01);
            }
            WindowEvent::MouseInput { button, state, .. } => match button {
                MouseButton::Left => self
                    .left
                    .handle_pressed(matches!(state, ElementState::Pressed)),
                MouseButton::Right => self.right.pressed = matches!(state, ElementState::Pressed),
                _ => {}
            },
            WindowEvent::CursorMoved { position, .. } => {
                let position = Vec2::new(position.x as f32, position.y as f32);
                self.right.new_position = position;
                self.left.handle_position(position);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let base = 0.1;
                let up = 1. + base;
                let down = 1. - base;
                let modifier = match delta {
                    glium::glutin::event::MouseScrollDelta::LineDelta(_, v) => {
                        if *v < 0. {
                            up
                        } else {
                            down
                        }
                    }
                    glium::glutin::event::MouseScrollDelta::PixelDelta(d) => {
                        if d.y < 0. {
                            up
                        } else {
                            down
                        }
                    }
                };
                self.camera.change_zoom(modifier);
            }
            _ => {}
        }
    }

    pub fn transform(&mut self) -> Result<(), String> {
        let selected = self.mesh.selected();
        if selected.is_empty() {
            return Err("No vertices selected".to_string());
        }

        let Matrices {
            combinatorial_laplacian,
            cotangent,
            geometric_laplacian,
            gradient_maps,
        } = self.model.differential_coordinates();

        let gradients: HashMap<usize, Vector3<f32>> = selected
            .iter()
            .flat_map(|&i| {
                let f = &self.model.faces[i];
                f.iter()
                    .map(|&v| (v, gradient_maps[i] * self.model.vertices[v].0))
                    .collect::<Vec<_>>()
            })
            .fold(HashMap::new(), |mut map, (v, f)| {
                *map.entry(v).or_default() += f;
                map
            });
        let new_gradients: HashMap<_, _> = gradients
            .iter()
            .map(|(&v, g)| (v, self.transformation * g))
            .collect();

        Ok(())
    }
}

pub fn start(model: super::Model) {
    let event_loop = glium::glutin::event_loop::EventLoopBuilder::with_user_event().build();
    let display = gui::create_display(&event_loop);

    let mesh = Mesh::load(&display, &model);

    let camera = Camera::new();

    let mut state = State {
        model,
        mesh,
        camera,
        right: Default::default(),
        left: Default::default(),
        resolution: Vec2::new(1., 1.),
        transformation: Matrix3::identity(),
        result: Ok(()),
    };

    gui::run(&mut state, display, event_loop)
}
