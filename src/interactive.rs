use std::collections::{HashMap, HashSet};

use egui::{Color32, Pos2, Rect, Rounding};
use glium::{
    backend::Facade,
    glutin::event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    Frame, Surface,
};

use nalgebra::{CsMatrix, DMatrix, Dim, Matrix3, VecStorage};
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

        let selected_vertices: HashSet<_> = selected
            .iter()
            .flat_map(|&f| self.model.faces[f].clone())
            .collect();

        let Matrices {
            combinatorial_laplacian,
            cotangent,
            geometric_laplacian,
            gradient_maps,
        } = self.model.differential_coordinates();

        let vertices: Vec<_> = self
            .model
            .vertices
            .iter()
            .enumerate()
            .filter_map(|(i, (v, _))| {
                if selected_vertices.contains(&i) {
                    Some((i, *v))
                } else {
                    None
                }
            })
            .collect();

        // Position of vertex in selected vector
        let indices: HashMap<_, _> = (0..vertices.len()).map(|i| (vertices[i].0, i)).collect();

        let row_idx: Vec<_> = (0..3 * selected.len()).flat_map(|i| [i; 3]).collect();
        let col_idx: Vec<_> = selected
            .iter()
            .flat_map(|&f| {
                self.model.faces[f]
                    .iter()
                    .map(|v| indices[v])
                    .flat_map(|i| [i; 3])
            })
            .collect();
        let values: Vec<_> = selected
            .iter()
            .flat_map(|&f| {
                (0..self.model.faces[f].len())
                    .flat_map(|i| {
                        gradient_maps[f]
                            .column(i)
                            .iter()
                            .copied()
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let g = CooMatrix::try_from_triplets(
            3 * selected.len(),
            selected_vertices.len(),
            row_idx,
            col_idx,
            values,
        )
        .unwrap();

        let g = CscMatrix::from(&g);

        let row_idx: Vec<_> = vertices
            .iter()
            .flat_map(|(v1, _)| {
                std::iter::repeat(indices[v1]).take(
                    cotangent[v1]
                        .keys()
                        .filter(|v2| selected_vertices.contains(v2))
                        .count(),
                )
            })
            .collect();
        let col_idx: Vec<_> = vertices
            .iter()
            .flat_map(|(v1, _)| cotangent[v1].keys().filter_map(|v2| indices.get(v2)))
            .copied()
            .collect();
        let values: Vec<_> = row_idx
            .iter()
            .zip(col_idx.iter())
            .map(|(v1, v2)| cotangent[&vertices[*v1].0][&vertices[*v2].0])
            .collect();

        // The cotangent represents the left-hand side of the eqaution, i.e. G^T*M_V*G
        let cotangent = CooMatrix::try_from_triplets(
            selected_vertices.len(),
            selected_vertices.len(),
            row_idx,
            col_idx,
            values,
        )
        .unwrap();

        let cotangent = CscMatrix::from(&cotangent);

        let gradients = selected.iter().flat_map(|f| {
            (0..3).into_iter().map(|i| {
                let v = self.model.vertices[self.model.faces[*f][i]].0;
                let map = gradient_maps[*f];
                map * v
            })
        });

        let g_tilde = gradients
            .map(|g| self.transformation * g)
            .collect::<Vec<_>>();

        let areas = self.model.areas();

        const TRIANGLE_VERTS: usize = 3;
        // Generate elements for M_v matrix
        let mut coo = CooMatrix::new(selected.len() * 3, selected.len() * 3);
        self.mesh
            .selected()
            .iter()
            .enumerate()
            .for_each(|(idx, tidx)| {
                (0..TRIANGLE_VERTS).for_each(|idx2| {
                    let fidx = idx * TRIANGLE_VERTS + idx2;
                    coo.push(fidx, fidx, areas[*tidx])
                })
            });

        let m_v = CscMatrix::from(&coo);

        let gtmv = g.transpose() * m_v;

        let g_tilde_x = g_tilde.iter().map(|g| g.x).collect::<Vec<_>>();
        let g_tilde_y = g_tilde.iter().map(|g| g.y).collect::<Vec<_>>();
        let g_tilde_z = g_tilde.iter().map(|g| g.z).collect::<Vec<_>>();
        let v_tilde_x = Self::solve_system(g_tilde_x, &gtmv, &cotangent);
        let v_tilde_y = Self::solve_system(g_tilde_y, &gtmv, &cotangent);
        let v_tilde_z = Self::solve_system(g_tilde_z, &gtmv, &cotangent);

        let v_tilde = v_tilde_x
            .into_iter()
            .zip(v_tilde_y.into_iter().zip(v_tilde_z.into_iter()))
            .map(|(x, (y, z))| Vec3::new(*x, *y, *z));

        selected_vertices
            .into_iter()
            .zip(v_tilde.into_iter())
            .for_each(|(idx, v)| {
                self.model.vertices[idx].0 = v;
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
