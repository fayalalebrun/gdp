use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, LineWriter, Write},
    iter,
};

use nalgebra::{Matrix3, Matrix4, Vector3, Vector4};
use obj::raw::{object::Polygon, parse_obj, RawObj};
use rand::seq::SliceRandom;

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

            println!(
                "from: {from_path}, to: {to_path}, output_path: {output_path}, n: {n}, k: {k}"
            );

            let from = read_obj(&from_path);
            let to = read_obj(&to_path);

            let output = icp_rigid_registration(&from, &to, n, k);

            let obj = output.to_obj();

            let file = File::create(output_path).unwrap();
            let mut file = LineWriter::new(file);

            for line in obj::raw::object::write_obj(&obj) {
                file.write_all(line.as_bytes()).unwrap();
                file.write_all(b"\n").unwrap();
            }
        }
        Some(com) => {
            println!("Unknown subcommand {com:?}");
        }
        None => {}
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

fn icp_rigid_registration(from: &Model, to: &Model, n: usize, k: f32) -> Model {
    let mut rng = &mut rand::thread_rng();

    let mut selected_points = from
        .vertices
        .choose_multiple(&mut rng, n)
        .map(|e| e.0)
        .collect::<Vec<_>>();

    let mut solution = Matrix4::identity();

    for _ in 0..1000 {
        let paired_closest = selected_points
            .iter()
            .filter_map(|s| {
                to.vertices
                    .iter()
                    .map(|v| v.0)
                    .min_by(|a, b| a.metric_distance(s).total_cmp(&b.metric_distance(s)))
                    .map(|o| (s, o))
            })
            .collect::<Vec<_>>();

        let distances = paired_closest
            .iter()
            .map(|(s, o)| s.metric_distance(&o))
            .collect::<Vec<_>>();

        let error: f32 = distances.iter().sum();

        println!("error: {error}");

        let median_distance = distances[distances.len() / 2];

        let paired_closest = paired_closest
            .into_iter()
            .zip(distances.into_iter())
            .filter_map(|((s, o), d)| {
                if d > k * median_distance {
                    None
                } else {
                    Some((*s, o))
                }
            })
            .collect();

        let theta = lst_solve(paired_closest);

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

fn lst_solve(pairs: Vec<(Vector3<f32>, Vector3<f32>)>) -> Matrix4<f32> {
    let from_centroid = pairs.iter().map(|e| e.0).sum::<Vector3<f32>>() / (pairs.len() as f32);
    let target_centroid = pairs.iter().map(|e| e.1).sum::<Vector3<f32>>() / (pairs.len() as f32);

    let pairs_centroid = pairs
        .into_iter()
        .map(|(from, target)| (from - from_centroid, target - target_centroid))
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

#[derive(Clone, Debug)]
struct Model {
    pub vertices: Vec<(Vector3<f32>, Vector3<f32>)>,
    pub faces: Vec<Vec<usize>>,
    pub edges: HashSet<[usize; 2]>,
    pub valences: HashMap<[usize; 2], usize>,
}

impl Model {
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
        self.faces
            .iter()
            .map(|f| {
                assert!(f.len() == 3, "Model should be triangulated");
                let random_origin = self.vertices[0].0;
                let (v1, _) = self.vertices[f[0]];
                let (v2, _) = self.vertices[f[1]];
                let (v3, _) = self.vertices[f[2]];
                let v1 = v1 - random_origin;
                let v2 = v2 - random_origin;
                let v3 = v3 - random_origin;

                let vol = v1.dot(&v2.cross(&v3)).abs() / 6.;

                vol
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
                        .map(|a| *a)
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

        return loops;
    }
}
