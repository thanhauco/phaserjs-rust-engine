//! Math utilities and types for the Phaser Rust Engine

pub mod vector;
pub mod matrix;
pub mod geom;
pub mod utils;

pub use vector::{Vector2, Vector3, Vector4};
pub use matrix::{Matrix3, Matrix4};
pub use geom::{AABB, Circle, Rectangle, Line, Polygon};
pub use utils::*;
