use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, LineWriter, Write},
    iter,
    str::FromStr,
};

use nalgebra::{DVector, Matrix3, Matrix4, Matrix6, Point3, Vector, Vector3, Vector4, Vector6};
use nalgebra_glm::{Mat4, Vec2};
use nalgebra_sparse::{CooMatrix, CscMatrix};
use obj::raw::{object::Polygon, parse_obj, RawObj};
use rand::seq::SliceRandom;

mod interactive;

fn main() {
    let mut pargs = pico_args::Arguments::from_env();
    match pargs.subcommand().unwrap().as_deref() {
        Some("analyze") => {
            let model_path = pargs.subcommand().unwrap().expect("Missing model path");
            let model = read_obj(&model_path);
            analyze_model(&model)
        }
        Some("icp") => {
            let from_path = pargs.subcommand().unwrap().expect("Missing 'from' model");
            let to_path = pargs.subcommand().unwrap().expect("Missing 'to' model");
            let output_path = pargs
                .value_from_str("--output-file")
                .unwrap_or("output.obj".to_string());
            let n = pargs
                .value_from_fn("--n", |n| n.parse::<usize>())
                .unwrap_or(10000);
            let k = pargs
                .value_from_fn("--k", |k| k.parse::<f32>())
                .unwrap_or(10.0);
            let distance = pargs
                .value_from_fn("--distance", |s| Distance::from_str(s))
                .unwrap_or(Distance::Point2Point);

            println!(
                "from: {from_path}, to: {to_path}, output_path: {output_path}, n: {n}, k: {k}"
            );

            let from = read_obj(&from_path);
            let to = read_obj(&to_path);

            let output = icp_rigid_registration(&from, &to, n, k, distance);

            let obj = output.to_obj();

            let file = File::create(output_path).unwrap();
            let mut file = LineWriter::new(file);

            for line in obj::raw::object::write_obj(&obj) {
                file.write_all(line.as_bytes()).unwrap();
                file.write_all(b"\n").unwrap();
            }
        }
        Some("gui") => {
            let model_path = pargs.subcommand().unwrap().expect("Missing model path");
            let model = read_obj(&model_path);
            interactive::start(model);
        }
        Some(com) => {
            println!("Unknown subcommand {com:?}");
        }
        None => {}
    }
}

#[derive(Clone, Debug)]
enum Distance {
    Point2Point,
    Point2Plane,
}

impl FromStr for Distance {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "point" => Ok(Self::Point2Point),
            "plane" => Ok(Self::Point2Plane),
            _ => panic!("Invalid distance measure"),
        }
    }
}

impl Distance {
    fn distance(&self, p1: Vector3<f32>, (p2, n): (Vector3<f32>, Vector3<f32>)) -> f32 {
        match self {
            Self::Point2Point => p1.metric_distance(&p2),
            Self::Point2Plane => (p1 - p2).dot(&n).abs(),
        }
    }
}

fn read_obj(path: &str) -> Model {
    let f = File::open(path).unwrap();
    let f = BufReader::new(f);

    let obj = parse_obj(f).unwrap();

    Model::from_obj(obj)
}

fn analyze_model(model: &Model) {
    let v = model.vertices.len() as i64;
    let e = model.edges.len() as i64;
    let f = model.faces.len() as i64;
    println!("Vertices: {}", v);
    println!("Edges: {}", e);
    println!("Faces: {}", f);
    println!("Genus: {}", (e - v - f) / 2 + 1);
    println!("Connected components: {}", model.connected_components());
    println!("Volume: {}", model.volume());
    println!("Boundary loops: {}", model.boundary_loops());
}

fn mat4_by_vec3(mat: Matrix4<f32>, vec: Vector3<f32>) -> Vector3<f32> {
    let res = mat * Vector4::new(vec.x, vec.y, vec.z, 1.0);
    Vector3::new(res.x, res.y, res.z)
}

fn icp_rigid_registration(from: &Model, to: &Model, n: usize, k: f32, distance: Distance) -> Model {
    let mut rng = &mut rand::thread_rng();

    let mut selected_points = from
        .vertices
        .choose_multiple(&mut rng, n)
        .map(|e| e.0)
        .collect::<Vec<_>>();

    let mut solution = Matrix4::identity();
    let mut patience_counter = 0;
    let mut prev_error = std::f32::INFINITY;

    for i in 0..1000 {
        let paired_closest = selected_points
            .iter()
            .filter_map(|s| {
                to.vertices
                    .iter()
                    .min_by(|a, b| a.0.metric_distance(s).total_cmp(&b.0.metric_distance(s)))
                    .map(|o| (s, o))
            })
            .collect::<Vec<_>>();

        let distances = paired_closest
            .iter()
            .map(|(&s, &o)| distance.distance(s, o))
            .collect::<Vec<_>>();

        let error: f32 = distances.iter().sum();

        println!("iter: {i} error: {error}");

        // Early stopping
        if prev_error - error < 1.0 {
            patience_counter += 1;
        } else {
            patience_counter = 0;
        }

        if patience_counter >= 5 {
            break;
        }

        prev_error = error;

        let median_distance = distances[distances.len() / 2];

        let paired_closest = paired_closest
            .into_iter()
            .zip(distances.into_iter())
            .filter_map(|((s, o), d)| {
                if d > k * median_distance {
                    None
                } else {
                    Some((*s, *o))
                }
            })
            .collect();

        let theta = match distance {
            Distance::Point2Point => lst_solve(paired_closest),
            Distance::Point2Plane => lrlst_solve(paired_closest),
        };

        selected_points
            .iter_mut()
            .for_each(|o| *o = mat4_by_vec3(theta, *o));
        solution = theta * solution;
    }

    let mut new = from.clone();

    new.vertices
        .iter_mut()
        .for_each(|(v, _)| *v = mat4_by_vec3(solution, *v));

    new
}

fn lst_solve(pairs: Vec<(Vector3<f32>, (Vector3<f32>, Vector3<f32>))>) -> Matrix4<f32> {
    let from_centroid = pairs.iter().map(|e| e.0).sum::<Vector3<f32>>() / (pairs.len() as f32);
    let target_centroid = pairs.iter().map(|e| e.1 .0).sum::<Vector3<f32>>() / (pairs.len() as f32);

    let pairs_centroid = pairs
        .into_iter()
        .map(|(from, target)| (from - from_centroid, target.0 - target_centroid))
        .collect::<Vec<_>>();

    let h: Matrix3<f32> = pairs_centroid
        .iter()
        .map(|(from, target)| from * target.transpose())
        .sum();

    let svd = h.svd(true, true);
    let x = svd.v_t.unwrap().transpose() * svd.u.unwrap().transpose();
    let r = if (x.determinant() - 1.0).abs() > 0.00001 {
        println!("lst solve failed with determinant {}", x.determinant());
        Matrix4::identity()
    } else {
        nalgebra_glm::mat3_to_mat4(&x)
    };
    nalgebra_glm::translate(&r, &(target_centroid - x * from_centroid))
}

fn lrlst_solve(pairs: Vec<(Vector3<f32>, (Vector3<f32>, Vector3<f32>))>) -> Matrix4<f32> {
    let cross: Vec<_> = pairs
        .iter()
        .map(|(p1, (_, n))| (p1.cross(n), n))
        .map(|(c, n)| Vector6::new(c.x, c.y, c.z, n.x, n.y, n.z))
        .collect();
    let dot: Vec<_> = pairs
        .iter()
        .map(|(p1, (p2, n))| (*p1 - *p2).dot(n))
        .collect();

    let a: Matrix6<_> = cross.iter().map(|v| v * v.transpose()).sum();
    let b: Vector6<_> = cross.iter().zip(dot.iter()).map(|(c, d)| *d * c).sum();

    let rt = a.try_inverse().expect("Unable to invert matrix") * -b;
    let r = Matrix3::new(1.0, -rt[2], rt[1], rt[2], 1.0, -rt[0], -rt[1], rt[0], 1.0);
    let t = Vector3::new(rt[3], rt[4], rt[5]);

    let svd = r.svd(true, true);
    let x = svd.u.unwrap() * svd.v_t.unwrap();
    let r = if (x.determinant() - 1.0).abs() > 0.00001 {
        println!("lrlst solve failed with determinant {}", x.determinant());
        Matrix4::identity()
    } else {
        nalgebra_glm::mat3_to_mat4(&x)
    };

    nalgebra_glm::translate(&r, &t)
}

/// Cross product of triangle vertices
/// Used for area and normal vector
fn triangle_cross(v1: Vector3<f32>, v2: Vector3<f32>, v3: Vector3<f32>) -> Vector3<f32> {
    (v2 - v1).cross(&(v3 - v1))
}

/// Normal = normalized(triangle_cross)
fn normal(v1: Vector3<f32>, v2: Vector3<f32>, v3: Vector3<f32>) -> Vector3<f32> {
    triangle_cross(v1, v2, v3).normalize()
}

/// Area = 0.5 * length(triangle_cross)
fn area(v1: Vector3<f32>, v2: Vector3<f32>, v3: Vector3<f32>) -> f32 {
    0.5 * triangle_cross(v1, v2, v3).norm()
}

#[derive(Debug)]
struct Matrices {
    /// Gradient matrix
    pub g: CscMatrix<f32>,
    /// Mass matrix
    pub m_v: CscMatrix<f32>,
    /// Selected vertices
    pub v: Vector3<DVector<f32>>,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub vertices: Vec<(Vector3<f32>, Vector3<f32>)>,
    pub faces: Vec<Vec<usize>>,
    pub edges: HashSet<[usize; 2]>,
    pub valences: HashMap<[usize; 2], usize>,
}

impl Model {
    pub fn from_parts(faces: Vec<Vec<usize>>, vertices: Vec<(Vector3<f32>, Vector3<f32>)>) -> Self {
        let valences = faces
            .iter()
            .flat_map(|f| {
                if f.len() >= 2 {
                    f.clone()
                        .into_iter()
                        .zip(f.clone().into_iter().skip(1).chain(iter::once(f[0])))
                        .map(|(v1, v2)| if v1 < v2 { [v1, v2] } else { [v2, v1] })
                        .collect()
                } else {
                    vec![]
                }
            })
            .fold(HashMap::new(), |mut acc, e| {
                *acc.entry(e).or_insert(0) += 1;
                acc
            });
        let edges = valences.keys().copied().collect();

        Self {
            vertices,
            faces,
            edges,
            valences,
        }
    }

    pub fn from_obj(model: RawObj) -> Self {
        let RawObj {
            positions,
            polygons,
            normals,
            ..
        } = model;

        let faces = polygons
            .into_iter()
            .map(|p| match p {
                obj::raw::object::Polygon::P(v) => v,
                obj::raw::object::Polygon::PT(v) => v.into_iter().map(|v| v.0).collect(),
                obj::raw::object::Polygon::PN(v) => v.into_iter().map(|v| v.0).collect(),
                obj::raw::object::Polygon::PTN(v) => v.into_iter().map(|v| v.0).collect(),
            })
            .collect::<Vec<_>>();

        assert!(
            positions.len() == normals.len(),
            "Assuming that each vertex has an exact correspondence to a normal"
        );

        let vertices = positions
            .into_iter()
            .zip(normals.into_iter())
            .map(|(v, n)| (Vector3::new(v.0, v.1, v.2), Vector3::new(n.0, n.1, n.2)))
            .collect();

        Self::from_parts(faces, vertices)
    }

    pub fn to_obj(&self) -> RawObj {
        let positions = self
            .vertices
            .iter()
            .map(|v| v.0)
            .map(|v| (v.x, v.y, v.z, 1.0))
            .collect();
        let normals = self
            .vertices
            .iter()
            .map(|v| v.1)
            .map(|v| (v.x, v.y, v.z))
            .collect();

        let polygons = self
            .faces
            .iter()
            .map(|f| Polygon::PN(f.iter().map(|i| (*i, *i)).collect()))
            .collect();

        RawObj {
            positions,
            normals,
            polygons,
            ..Default::default()
        }
    }

    pub fn volume(&self) -> f32 {
        let o = self.vertices[0].0;
        self.faces
            .iter()
            .map(|f| {
                assert!(f.len() == 3, "Model should be triangulated");
                let (v1, _) = self.vertices[f[0]];
                let (v2, _) = self.vertices[f[1]];
                let (v3, _) = self.vertices[f[2]];
                if v1 == o || v2 == o || v3 == o {
                    return 0.0;
                }
                let v1 = v1 - o;
                let v2 = v2 - o;
                let v3 = v3 - o;

                v1.dot(&v2.cross(&v3)).abs() / 6.
            })
            .sum::<f32>()
            .abs()
    }

    pub fn connected_components(&self) -> u64 {
        let mut connections: HashMap<usize, HashSet<usize>> = HashMap::new();
        for [e1, e2] in self.edges.iter() {
            let mut add = |from, to| {
                if !connections.contains_key(from) {
                    connections.insert(*from, HashSet::new());
                }
                connections.get_mut(from).unwrap().insert(to);
            };
            add(e1, *e2);
            add(e2, *e1);
        }

        let mut available: HashSet<_> = (0..self.vertices.len()).collect();

        let mut components = 0;
        while !available.is_empty() {
            components += 1;
            let next = *available.iter().next().unwrap();
            let mut queue = vec![next];

            while let Some(next) = queue.pop() {
                available.remove(&next);

                if let Some(outgoing) = connections.get(&next) {
                    let mut outgoing = outgoing
                        .iter()
                        .to_owned()
                        .filter(|a| available.contains(a))
                        .copied()
                        .collect();
                    queue.append(&mut outgoing);
                }
            }
        }
        components
    }

    pub fn boundary_loops(&self) -> u64 {
        let mut boundary_vertices: HashSet<_> = self
            .valences
            .iter()
            .filter(|(_, &v)| v == 1)
            .flat_map(|(e, _)| e)
            .copied()
            .collect();
        let mut neighbors: HashMap<usize, Vec<usize>> = HashMap::new();

        for [e1, e2] in self.edges.iter() {
            if boundary_vertices.contains(e1) && boundary_vertices.contains(e2) {
                neighbors.entry(*e1).or_default().push(*e2);
                neighbors.entry(*e2).or_default().push(*e1);
            }
        }

        let mut loops = 0;

        while !boundary_vertices.is_empty() {
            let vertex = *boundary_vertices.iter().next().unwrap();

            let mut stack = vec![vertex];

            while let Some(vertex) = stack.pop() {
                boundary_vertices.remove(&vertex);

                for neighbor in neighbors.get(&vertex).unwrap_or(&Vec::new()) {
                    if boundary_vertices.contains(neighbor) {
                        stack.push(*neighbor);
                    }
                }
            }

            loops += 1;
        }

        loops
    }

    /// Returns a mask with the triangles contained by the rectangle.
    pub fn triangles_in_rect(&self, mvp: Mat4, bottom_left: Vec2, top_right: Vec2) -> Vec<bool> {
        self.faces
            .iter()
            .map(|t| {
                let v1 = self.vertices[t[0]].0;
                let v2 = self.vertices[t[1]].0;
                let v3 = self.vertices[t[2]].0;

                let center = (v1 + v2 + v3) / 3.0; // Center of triangle

                center
            })
            .map(|v| mvp.transform_point(&Point3::from(v)))
            .map(|v| Vec2::new(v.x, v.y))
            .map(|v| {
                if (v - bottom_left).dot(&(v - top_right)) < 0. {
                    true
                } else {
                    false
                }
            })
            .collect()
    }

    /// Gradient matrix for a given face/triangle
    /// Multiply this with the outcome of a linear polynomial applied on the vertices of the face
    fn gradient_map(&self, face: usize) -> Matrix3<f32> {
        let face = &self.faces[face];
        assert!(face.len() == 3, "Model should be triangulated");

        let v = [
            self.vertices[face[0]].0,
            self.vertices[face[1]].0,
            self.vertices[face[2]].0,
        ];
        let e = [v[2] - v[1], v[0] - v[2], v[1] - v[0]];

        // This is the triangle normal, which is distinct from the vertex normals
        let n = normal(v[0], v[1], v[2]);
        let half_inv_area = 0.5 / area(v[0], v[1], v[2]);

        half_inv_area * Matrix3::from_columns(&[n.cross(&e[0]), n.cross(&e[1]), n.cross(&e[2])])
    }

    /// Area of each face/triangle
    fn areas(&self, selected_faces: &Vec<usize>) -> Vec<f32> {
        selected_faces
            .iter()
            .map(|&f| {
                let f = &self.faces[f];
                area(
                    self.vertices[f[0]].0,
                    self.vertices[f[1]].0,
                    self.vertices[f[2]].0,
                )
            })
            .collect()
    }

    /// Collection of faces each vertex belongs to
    fn vertex_faces(&self, selected_faces: &Vec<usize>) -> HashMap<usize, Vec<usize>> {
        selected_faces
            .iter()
            .enumerate()
            .flat_map(|(i, &f)| self.faces[f].iter().map(|v| (*v, i)).collect::<Vec<_>>())
            .fold(HashMap::new(), |mut map, (v, f)| {
                map.entry(v).or_default().push(f);
                map
            })
    }

    /// For each vertex, keep track of the gradient product sum for each neighboring vertex
    fn cotangent(
        &self,
        vertex_faces: &HashMap<usize, Vec<usize>>,
        gradient_maps: &[Matrix3<f32>],
        areas: &[f32],
    ) -> HashMap<usize, HashMap<usize, f32>> {
        vertex_faces
            .iter()
            .map(|(&v, faces)| {
                (
                    v,
                    faces
                        .iter()
                        .flat_map(|&f| {
                            let gradient = gradient_maps[f]
                                .column(self.faces[f].iter().position(|&o| o == v).unwrap());
                            (0..self.faces[f].len())
                                .map(|i| {
                                    (
                                        self.faces[f][i],
                                        areas[f] * gradient_maps[f].column(i).dot(&gradient),
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                        .fold(HashMap::new(), |mut map, (v, f)| {
                            *map.entry(v).or_default() += f;
                            map
                        }),
                )
            })
            .collect()
    }

    /// The neighbor vertices for each vertex
    /// This can be used for computing the combinatorial laplacian
    fn neighbors(&self, vertex_faces: &HashMap<usize, Vec<usize>>) -> HashMap<usize, Vec<usize>> {
        vertex_faces
            .iter()
            .map(|(&v, faces)| {
                (
                    v,
                    faces
                        .iter()
                        .flat_map(|&f| {
                            self.faces[f]
                                .iter()
                                .filter(|&&o| o != v)
                                .collect::<Vec<_>>()
                        })
                        .copied()
                        .collect(),
                )
            })
            .collect()
    }

    /// Combined area of each triangle a vertex belongs to
    fn vertex_areas(
        &self,
        vertex_faces: &HashMap<usize, Vec<usize>>,
        areas: &[f32],
    ) -> HashMap<usize, f32> {
        vertex_faces
            .iter()
            .map(|(&v, faces)| (v, faces.iter().map(|&f| areas[f]).sum()))
            .collect()
    }

    fn geometric_laplacian(
        &self,
        combined_area: &HashMap<usize, f32>,
        cotangent: &HashMap<usize, HashMap<usize, f32>>,
    ) -> HashMap<usize, HashMap<usize, f32>> {
        combined_area
            .iter()
            .map(|(v, &a)| {
                (
                    *v,
                    cotangent[v]
                        .iter()
                        .map(|(&o, &f)| (o, a * f))
                        .collect::<HashMap<usize, f32>>(),
                )
            })
            .collect()
    }

    fn differential_coordinates(&self, selected_faces: &Vec<usize>) -> Matrices {
        let selected_vertices: HashSet<_> = selected_faces
            .iter()
            .flat_map(|&f| self.faces[f].clone())
            .collect();

        let vertices: Vec<_> = self
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

        let indices: HashMap<_, _> = (0..vertices.len()).map(|i| (vertices[i].0, i)).collect();
        let gradient_maps: Vec<_> = selected_faces
            .iter()
            .map(|&f| self.gradient_map(f))
            .collect();
        let areas = self.areas(selected_faces);

        let mut g = CooMatrix::new(selected_faces.len() * 3, selected_vertices.len());
        selected_faces.iter().enumerate().for_each(|(i, &f)| {
            (0..self.faces[f].len()).for_each(|v| {
                (0..3).for_each(|d| {
                    g.push(
                        i * 3 + d,
                        indices[&self.faces[f][v]],
                        gradient_maps[i][(d, v)],
                    )
                });
            })
        });

        let g = CscMatrix::from(&g);

        let mut coo = CooMatrix::new(selected_faces.len() * 3, selected_faces.len() * 3);
        selected_faces.iter().enumerate().for_each(|(i, &f)| {
            (0..self.faces[f].len()).for_each(|v| {
                let fidx = i * 3 + v;
                coo.push(fidx, fidx, areas[i])
            })
        });

        let m_v = CscMatrix::from(&coo);

        let v =
            Vector3::from_iterator((0..3).map(|i| {
                DVector::from_iterator(vertices.len(), vertices.iter().map(|(_, v)| v[i]))
            }));

        // let vertex_faces = self.vertex_faces(selected_faces);
        // let cotangent = self.cotangent(&vertex_faces, &gradient_maps, &areas);
        // let combinatorial_laplacian = self.neighbors(&vertex_faces);
        // let vertex_areas = self.vertex_areas(&vertex_faces, &areas);
        // let geometric_laplacian = self.geometric_laplacian(&vertex_areas, &cotangent);

        Matrices { g, m_v, v }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn build_tetrahedron() -> Model {
        let v = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 3.0_f32.sqrt(), 0.0),
            Vector3::new(0.0, 3.0_f32.sqrt() / 3.0, (8.0 / 3.0_f32).sqrt()),
        ];
        Model::from_parts(
            vec![vec![0, 1, 2], vec![2, 1, 3], vec![0, 2, 3], vec![1, 0, 3]],
            v.iter().map(|v| (*v, Vector3::zeros())).collect(),
        )
    }

    fn build_unequal_tetrahedron() -> Model {
        let v = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 3.0_f32.sqrt(), 0.0),
            Vector3::new(0.0, 3.0_f32.sqrt() / 3.0, 2.0),
        ];
        Model::from_parts(
            vec![vec![0, 1, 2], vec![2, 1, 3], vec![0, 2, 3], vec![1, 0, 3]],
            v.iter().map(|v| (*v, Vector3::zeros())).collect(),
        )
    }

    #[test]
    fn triangle_normal_test() {
        // Polarity test
        let v = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
        ];
        let n = normal(v[0], v[1], v[2]);
        assert_relative_eq!(n, Vector3::new(0.0, 0.0, 1.0));
        let n = normal(v[0], v[2], v[1]);
        assert_relative_eq!(n, Vector3::new(0.0, 0.0, -1.0));

        // 45 degree slope
        let v = [
            Vector3::new(1.0, 0.0, 2.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(-1.0, 0.0, 2.0),
        ];
        let n = normal(v[0], v[1], v[2]);
        assert_relative_eq!(
            n,
            Vector3::new(0.0, 1.0, 1.0) * 0.5 * std::f32::consts::SQRT_2
        );
    }

    #[test]
    fn triangle_area_test() {
        let v = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
        ];
        let a = area(v[0], v[1], v[2]);
        assert_relative_eq!(a, 0.5 * 2.0 * 2.0);
        let v = [
            Vector3::new(1.0, 0.0, 2.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(-1.0, 0.0, 2.0),
        ];
        let a = area(v[0], v[1], v[2]);
        assert_relative_eq!(a, 0.5 * 2.0 * 8.0_f32.sqrt());
    }

    #[test]
    fn gradient_map_test() {
        let v = [
            Vector3::new(1.0, 0.0, 2.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::new(-1.0, 0.0, 2.0),
        ];
        let e = [
            Vector3::new(-1.0, -2.0, 2.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(-1.0, 2.0, -2.0),
        ];
        let n = Vector3::new(0.0, 1.0, 1.0) * 0.5 * std::f32::consts::SQRT_2;
        let a = 0.5 * 2.0 * 8.0_f32.sqrt();

        let mesh = Model::from_parts(vec![vec![0, 1, 2]], vec![(v[0], n), (v[1], n), (v[2], n)]);

        let polynomial = |v: &Vector3<f32>| -> f32 { v.x + v.y * 2.0 };
        let u = Vector3::from_iterator(v.iter().map(|v| polynomial(v)));

        let gradients = mesh.gradient_map(0) * u;
        let c: Vec<_> = e
            .iter()
            .zip(u.iter())
            .map(|(e, &u)| u * n.cross(e))
            .collect();
        assert_relative_eq!(gradients, 0.5 / a * (c[0] + c[1] + c[2]));
    }

    #[test]
    fn mesh_areas_test() {
        let tetrahedron = build_tetrahedron();
        let a = 0.5 * 2.0 * 3.0_f32.sqrt();
        tetrahedron
            .areas(&vec![0, 1, 2, 3])
            .into_iter()
            .for_each(|f| assert_relative_eq!(f, a));

        let tetrahedron = build_unequal_tetrahedron();
        let a = 0.5 * 2.0 * 3.0_f32.sqrt();
        let areas = tetrahedron.areas(&vec![0, 1, 2, 3]);
        assert_relative_eq!(areas[0], a);

        let a = 0.5 * 2.0 * (13.0 / 3.0_f32).sqrt();
        areas
            .into_iter()
            .skip(1)
            .for_each(|f| assert_relative_eq!(f, a));
    }

    #[test]
    fn mesh_vertex_faces_test() {
        let tetrahedron = build_tetrahedron();
        let vertex_faces = tetrahedron.vertex_faces(&vec![0, 1, 2, 3]);
        vertex_faces.values().for_each(|f| assert_eq!(f.len(), 3));
        assert_eq!(vertex_faces[&0], vec![0, 2, 3]);
        assert_eq!(vertex_faces[&1], vec![0, 1, 3]);
        assert_eq!(vertex_faces[&2], vec![0, 1, 2]);
        assert_eq!(vertex_faces[&3], vec![1, 2, 3]);
    }

    #[test]
    fn mesh_cotangent_test() {
        let tetrahedron = build_tetrahedron();
        let Matrices { cotangent, .. } = tetrahedron.differential_coordinates(&vec![0, 1, 2, 3]);
        let mut matrix = Matrix4::zeros();

        for i in 0..4 {
            for j in 0..4 {
                assert!(cotangent[&i].contains_key(&j));
                assert_relative_eq!(cotangent[&i][&j], cotangent[&j][&i]);
                matrix[(i, j)] = cotangent[&i][&j];
            }
        }

        // Assert strange defintion of non-negative matrix
        let v = Vector4::new(1.0, 2.0, 3.0, 4.0);
        assert!((v.transpose() * matrix * v)[0] > -std::f32::EPSILON);
    }

    #[test]
    fn mesh_combinatorial_laplacian_test() {
        let tetrahedron = build_tetrahedron();
        let Matrices {
            combinatorial_laplacian,
            ..
        } = tetrahedron.differential_coordinates(&vec![0, 1, 2, 3]);

        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    continue;
                }
                assert!(combinatorial_laplacian[&i].contains(&j));
            }
        }
    }

    #[test]
    fn mesh_geometric_laplacian_test() {
        let tetrahedron = build_tetrahedron();
        let Matrices {
            geometric_laplacian,
            ..
        } = tetrahedron.differential_coordinates(&vec![0, 1, 2, 3]);

        let mut matrix = Matrix4::zeros();
        for i in 0..4 {
            for j in 0..4 {
                matrix[(i, j)] = geometric_laplacian[&i][&j];
            }
        }

        // Assert non-negative eigenvalues
        matrix
            .eigenvalues()
            .unwrap()
            .into_iter()
            .for_each(|&e| assert!(e > -std::f32::EPSILON));
    }
}
