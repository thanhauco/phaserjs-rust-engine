//! Math utility functions

use std::f32::consts::PI;

/// Linear interpolation between two values
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Clamp a value between min and max
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Convert degrees to radians
pub fn deg_to_rad(degrees: f32) -> f32 {
    degrees * PI / 180.0
}

/// Convert radians to degrees
pub fn rad_to_deg(radians: f32) -> f32 {
    radians * 180.0 / PI
}

/// Normalize an angle to the range [0, 2π)
pub fn normalize_angle(angle: f32) -> f32 {
    let two_pi = 2.0 * PI;
    let mut normalized = angle % two_pi;
    if normalized < 0.0 {
        normalized += two_pi;
    }
    normalized
}

/// Calculate the shortest angular distance between two angles
pub fn angle_distance(a: f32, b: f32) -> f32 {
    let diff = (b - a) % (2.0 * PI);
    if diff > PI {
        diff - 2.0 * PI
    } else if diff < -PI {
        diff + 2.0 * PI
    } else {
        diff
    }
}

/// Smooth step interpolation
pub fn smooth_step(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Check if two floats are approximately equal
pub fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() < epsilon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp() {
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_deg_to_rad() {
        assert!(approx_eq(deg_to_rad(0.0), 0.0, 0.001));
        assert!(approx_eq(deg_to_rad(90.0), PI / 2.0, 0.001));
        assert!(approx_eq(deg_to_rad(180.0), PI, 0.001));
    }

    #[test]
    fn test_normalize_angle() {
        assert!(approx_eq(normalize_angle(0.0), 0.0, 0.001));
        assert!(approx_eq(normalize_angle(2.0 * PI), 0.0, 0.001));
        assert!(approx_eq(normalize_angle(-PI), PI, 0.001));
    }
}
