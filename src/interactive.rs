use std::collections::{HashMap, HashSet};

use egui::{Color32, Pos2, Rect, Rounding};
use glium::{
    backend::Facade,
    glutin::event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    Frame, Surface,
};

use nalgebra::{CsMatrix, DMatrix, DVector, Dim, Matrix3, VecStorage, Vector3};
use nalgebra_glm::{Mat4, Vec2, Vec3};
use nalgebra_sparse::{factorization::CscCholesky, CooMatrix, CscMatrix};

use crate::{area, Matrices, Model};

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
    refresh_mesh: bool,
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
            });

            if ui.button("Apply").clicked() {
                self.result = self.transform();
            }

            ui.separator();

            if ui.button("Smooth Combinatorial").clicked() {
                self.smooth_combinatorial(0.5);
            }

            ui.separator();

            if ui.button("Smooth Geometric").clicked() {
                self.smooth_geometric(0.5);
            }

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
        if self.refresh_mesh {
            self.refresh_mesh = false;
            self.mesh = Mesh::load(facade, &self.model);
        }

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

        let selected_vertices = selected
            .iter()
            .flat_map(|&f| self.model.faces[f].clone())
            .collect::<HashSet<_>>();

        let triangle_idxs = (0..self.model.faces.len()).collect::<Vec<_>>();
        let vert_idxs = (0..self.model.vertices.len()).collect::<Vec<_>>();

        let Matrices { g, m_v, v, m } = self
            .model
            .differential_coordinates(&triangle_idxs, &vert_idxs);

        let g_x = &g * &v.x;
        let g_y = &g * &v.y;
        let g_z = &g * &v.z;

        let g_tilde = DVector::from_iterator(
            g_x.len(),
            (0..g_x.len()).map(|i| {
                if selected_vertices.contains(&i) {
                    self.transformation * Vector3::new(g_x[i], g_y[i], g_z[i])
                } else {
                    Vector3::new(g_x[i], g_y[i], g_z[i])
                }
            }),
        );
        let g_tilde_x = g_tilde.iter().map(|g| g.x).collect::<Vec<_>>();
        let g_tilde_y = g_tilde.iter().map(|g| g.y).collect::<Vec<_>>();
        let g_tilde_z = g_tilde.iter().map(|g| g.z).collect::<Vec<_>>();

        let gtmv = &g.transpose() * &m_v;

        let cotangent = (&gtmv * &g) + 0.001 * m;

        let v_tilde_x = Self::solve_system(g_tilde_x, &gtmv, &cotangent);
        let v_tilde_y = Self::solve_system(g_tilde_y, &gtmv, &cotangent);
        let v_tilde_z = Self::solve_system(g_tilde_z, &gtmv, &cotangent);

        let v_tilde: Vec<_> = v_tilde_x
            .into_iter()
            .zip(v_tilde_y.into_iter().zip(v_tilde_z.into_iter()))
            .map(|(x, (y, z))| Vec3::new(*x, *y, *z))
            .collect();

        let center = vert_idxs
            .iter()
            .map(|&v| self.model.vertices[v].0)
            .sum::<Vector3<_>>()
            / self.model.vertices.len() as f32;
        let center_tilde = v_tilde.iter().sum::<Vector3<_>>() / self.model.vertices.len() as f32;

        let offset = center - center_tilde;

        vert_idxs.into_iter().zip(v_tilde).for_each(|(idx, v)| {
            self.model.vertices[idx].0 = v + offset;
        });

        self.refresh_mesh = true;

        Ok(())
    }

    pub fn solve_system(
        g_tilde: Vec<f32>,
        gtmv: &CscMatrix<f32>,
        cotangent: &CscMatrix<f32>,
    ) -> DMatrix<f32> {
        let g_tilde = DMatrix::from_vec_storage(VecStorage::new(
            Dim::from_usize(g_tilde.len()),
            Dim::from_usize(1),
            g_tilde,
        ));

        let b = gtmv * g_tilde;
        let a = CscCholesky::factor(cotangent).unwrap();
        a.solve(&b)
    }

    pub fn smooth_combinatorial(&mut self, factor: f32) {
        let mut selected = self.mesh.selected().clone();

        if selected.is_empty() {
            selected = (0..self.model.faces.len()).collect::<Vec<_>>();
        }

        let selected_faces: Vec<_> = (0..self.model.faces.len()).collect();
        let neighbors = self.model.neighbors(&selected_faces);
        let n = neighbors.len();

        let vertices: Vec<_> = self
            .model
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if neighbors.contains_key(&i) {
                    Some((i, v.0))
                } else {
                    None
                }
            })
            .collect();

        let indices: HashMap<_, _> = (0..vertices.len()).map(|i| (vertices[i].0, i)).collect();

        let v_x = DVector::from_iterator(vertices.len(), vertices.iter().map(|v| v.1.x));
        let v_y = DVector::from_iterator(vertices.len(), vertices.iter().map(|v| v.1.y));
        let v_z = DVector::from_iterator(vertices.len(), vertices.iter().map(|v| v.1.z));

        let mut cl = CooMatrix::new(n, n);
        vertices.iter().for_each(|(v1, _)| {
            let i = indices[v1];
            cl.push(i, i, 1.0);
            let deg = neighbors[v1]
                .iter()
                .filter(|v2| indices.contains_key(v2))
                .count();
            neighbors[v1]
                .iter()
                .filter(|v2| indices.contains_key(v2))
                .for_each(|v2| cl.push(i, indices[v2], -1.0 / deg as f32))
        });

        let cl = CscMatrix::from(&cl);

        let v_tilde_x = &cl * v_x;
        let v_tilde_y = &cl * v_y;
        let v_tilde_z = &cl * v_z;

        let v_tilde: Vec<_> = v_tilde_x
            .into_iter()
            .zip(v_tilde_y.into_iter().zip(v_tilde_z.into_iter()))
            .map(|(x, (y, z))| Vec3::new(*x, *y, *z))
            .collect();

        let selected_vertices = selected
            .iter()
            .flat_map(|&f| self.model.faces[f].clone())
            .collect::<HashSet<_>>();

        vertices
            .iter()
            .zip(v_tilde.iter())
            .filter(|((i, _), _)| selected_vertices.contains(i))
            .for_each(|((i, _), v)| {
                self.model.vertices[*i].0 -= factor * v;
            });

        self.refresh_mesh = true;
    }

    pub fn smooth_geometric(&mut self, factor: f32) {
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
        refresh_mesh: false,
    };

    gui::run(&mut state, display, event_loop)
}
