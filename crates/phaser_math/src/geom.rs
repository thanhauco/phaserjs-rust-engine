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
    
    pub fn area(&self) -> f32 {
        std::f32::consts::PI * self.radius * self.radius
    }
}

/// Rectangle shape
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rectangle {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rectangle {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }
    
    pub fn from_aabb(aabb: AABB) -> Self {
        Self {
            x: aabb.min.x,
            y: aabb.min.y,
            width: aabb.width(),
            height: aabb.height(),
        }
    }
    
    pub fn to_aabb(&self) -> AABB {
        AABB {
            min: Vector2::new(self.x, self.y),
            max: Vector2::new(self.x + self.width, self.y + self.height),
        }
    }
    
    pub fn center(&self) -> Vector2 {
        Vector2::new(self.x + self.width * 0.5, self.y + self.height * 0.5)
    }
    
    pub fn contains_point(&self, point: Vector2) -> bool {
        point.x >= self.x
            && point.x <= self.x + self.width
            && point.y >= self.y
            && point.y <= self.y + self.height
    }
    
    pub fn intersects(&self, other: &Rectangle) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
    
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

/// Line segment
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Line {
    pub start: Vector2,
    pub end: Vector2,
}

impl Line {
    pub fn new(start: Vector2, end: Vector2) -> Self {
        Self { start, end }
    }
    
    pub fn length(&self) -> f32 {
        self.start.distance(self.end)
    }
    
    pub fn length_squared(&self) -> f32 {
        self.start.distance_squared(self.end)
    }
    
    pub fn direction(&self) -> Vector2 {
        (self.end - self.start).normalize()
    }
    
    pub fn point_at(&self, t: f32) -> Vector2 {
        self.start.lerp(self.end, t)
    }
}

/// Polygon shape
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Polygon {
    pub points: Vec<Vector2>,
}

impl Polygon {
    pub fn new(points: Vec<Vector2>) -> Self {
        Self { points }
    }
    
    pub fn contains_point(&self, point: Vector2) -> bool {
        if self.points.len() < 3 {
            return false;
        }
        
        let mut inside = false;
        let mut j = self.points.len() - 1;
        
        for i in 0..self.points.len() {
            let pi = self.points[i];
            let pj = self.points[j];
            
            if ((pi.y > point.y) != (pj.y > point.y))
                && (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x)
            {
                inside = !inside;
            }
            
            j = i;
        }
        
        inside
    }
    
    pub fn bounds(&self) -> AABB {
        if self.points.is_empty() {
            return AABB::new(Vector2::ZERO, Vector2::ZERO);
        }
        
        let mut min = self.points[0];
        let mut max = self.points[0];
        
        for point in &self.points[1..] {
            min = min.min(*point);
            max = max.max(*point);
        }
        
        AABB::new(min, max)
    }
}
