//! Geometric primitives

use crate::Vector2;
use serde::{Deserialize, Serialize};

/// Axis-Aligned Bounding Box
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AABB {
    pub min: Vector2,
    pub max: Vector2,
}

impl AABB {
    pub fn new(min: Vector2, max: Vector2) -> Self {
        Self { min, max }
    }
    
    pub fn from_center_size(center: Vector2, size: Vector2) -> Self {
        let half_size = size * 0.5;
        Self {
            min: center - half_size,
            max: center + half_size,
        }
    }
    
    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }
    
    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }
    
    pub fn center(&self) -> Vector2 {
        (self.min + self.max) * 0.5
    }
    
    pub fn contains_point(&self, point: Vector2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }
    
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }
}

/// Circle shape
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Circle {
    pub center: Vector2,
    pub radius: f32,
}

impl Circle {
    pub fn new(center: Vector2, radius: f32) -> Self {
        Self { center, radius }
    }
    
    pub fn contains_point(&self, point: Vector2) -> bool {
        self.center.distance_squared(point) <= self.radius * self.radius
    }
    
    pub fn intersects(&self, other: &Circle) -> bool {
        let distance_squared = self.center.distance_squared(other.center);
        let radius_sum = self.radius + other.radius;
        distance_squared <= radius_sum * radius_sum
    }
}
