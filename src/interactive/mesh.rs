use glium::{
    backend::Facade, implement_vertex, index::PrimitiveType, uniform, BackfaceCullingMode,
    DrawParameters, IndexBuffer, Program, Surface, VertexBuffer,
};
use nalgebra::Point3;
use nalgebra_glm::Mat4;

use crate::Model;

#[derive(Copy, Clone)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

implement_vertex!(Vertex, position, normal);

pub struct Mesh {
    vertices: VertexBuffer<Vertex>,
    indices: IndexBuffer<u32>,
    selected_indices: IndexBuffer<u32>,
    program: Program,
}

impl Mesh {
    pub fn load(facade: &dyn Facade, model: &Model) -> Self {
        let vertices = model
            .vertices
            .iter()
            .map(|(v, n)| {
                let position: [f32; 3] = (*v).into();
                let normal: [f32; 3] = (*n).into();
                Vertex { position, normal }
            })
            .collect::<Vec<_>>();

        let indices = model
            .faces
            .iter()
            .flat_map(|f| {
                assert!(f.len() == 3, "Model is not triangulated");

                f.iter().map(|u| *u as u32)
            })
            .collect::<Vec<_>>();

        let vertices = VertexBuffer::new(facade, &vertices).unwrap();
        let indices = IndexBuffer::new(facade, PrimitiveType::TrianglesList, &indices).unwrap();

        let selected_indices =
            IndexBuffer::new(facade, PrimitiveType::TrianglesList, &[0u32; 0]).unwrap();

        let program = Program::from_source(
            facade,
            include_str!("vert.glsl"),
            include_str!("frag.glsl"),
            None,
        )
        .unwrap();

        Self {
            vertices,
            indices,
            selected_indices,
            program,
        }
    }

    pub fn set_selected(&mut self, facade: &dyn Facade, model: &Model, selected: &[bool]) {
        let indices = model
            .faces
            .iter()
            .enumerate()
            .filter_map(|(i, f)| if selected[i] { None } else { Some(f) })
            .flat_map(|f| f.iter().map(|u| *u as u32))
            .collect::<Vec<_>>();

        let selected_indices = model
            .faces
            .iter()
            .enumerate()
            .filter_map(|(i, f)| if !selected[i] { None } else { Some(f) })
            .flat_map(|f| f.iter().map(|u| *u as u32))
            .collect::<Vec<_>>();

        self.indices = IndexBuffer::new(facade, PrimitiveType::TrianglesList, &indices).unwrap();

        self.selected_indices =
            IndexBuffer::new(facade, PrimitiveType::TrianglesList, &selected_indices).unwrap();
    }

    pub fn draw<S: Surface>(
        &self,
        surface: &mut S,
        model: Mat4,
        view: Mat4,
        projection: Mat4,
        view_pos: Point3<f32>,
    ) {
        let projection: [[f32; 4]; 4] = projection.into();
        let view: [[f32; 4]; 4] = view.into();
        let model: [[f32; 4]; 4] = model.into();
        let view_pos: [f32; 3] = view_pos.into();

        for (c, indices) in [
            ([1.0f32; 3], &self.selected_indices),
            ([0.0, 0.0, 1.0], &self.indices),
        ]
        .into_iter()
        {
            let uniforms = uniform! {
            projection: projection,
            view: view,
            model: model,
            viewPos: view_pos,
            color: c
                };
            surface
                .draw(
                    &self.vertices,
                    indices,
                    &self.program,
                    &uniforms,
                    &DrawParameters {
                        depth: glium::Depth {
                            test: glium::DepthTest::IfLess,
                            write: true,
                            ..Default::default()
                        },
                        backface_culling: BackfaceCullingMode::CullCounterClockwise,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }
}
