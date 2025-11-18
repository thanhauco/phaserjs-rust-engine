//! Property-based tests for math operations

use phaser_math::{Matrix3, Vector2};
use proptest::prelude::*;
use std::f32::consts::PI;

// **Feature: phaser-rust-engine, Property 15: Transform matrix consistency**
proptest! {
    #[test]
    fn test_transform_matrix_consistency(
        x in -1000.0f32..1000.0,
        y in -1000.0f32..1000.0,
        rotation in 0.0f32..(2.0 * PI),
        scale_x in 0.1f32..10.0,
        scale_y in 0.1f32..10.0,
    ) {
        // Create a transform matrix from translation, rotation, and scale
        let translation = Matrix3::from_translation(Vector2::new(x, y));
        let rotation_mat = Matrix3::from_angle(rotation);
        let scale = Matrix3::from_scale(Vector2::new(scale_x, scale_y));
        
        // Combined transformation: scale -> rotate -> translate
        let combined = translation * rotation_mat * scale;
        
        // Apply to a test point
        let test_point = Vector2::new(1.0, 0.0);
        let transformed = combined.transform_point2(test_point);
        
        // The transformed point should be finite
        prop_assert!(transformed.x.is_finite());
        prop_assert!(transformed.y.is_finite());
        
        // Verify the transformation is consistent
        // If we apply the inverse, we should get back close to the original
        if let Some(inverse) = combined.inverse() {
            let back = inverse.transform_point2(transformed);
            let epsilon = 0.01;
            prop_assert!((back.x - test_point.x).abs() < epsilon);
            prop_assert!((back.y - test_point.y).abs() < epsilon);
        }
    }
    
    #[test]
    fn test_matrix_multiplication_associativity(
        a_x in -100.0f32..100.0,
        a_y in -100.0f32..100.0,
        b_angle in 0.0f32..(2.0 * PI),
        c_scale in 0.1f32..10.0,
    ) {
        let a = Matrix3::from_translation(Vector2::new(a_x, a_y));
        let b = Matrix3::from_angle(b_angle);
        let c = Matrix3::from_scale(Vector2::new(c_scale, c_scale));
        
        // Matrix multiplication should be associative: (A * B) * C = A * (B * C)
        let left = (a * b) * c;
        let right = a * (b * c);
        
        let epsilon = 0.001;
        for i in 0..3 {
            for j in 0..3 {
                let diff = (left.col(i)[j] - right.col(i)[j]).abs();
                prop_assert!(diff < epsilon, "Matrix multiplication not associative at ({}, {}): diff = {}", i, j, diff);
            }
        }
    }
    
    #[test]
    fn test_rotation_matrix_properties(
        angle in 0.0f32..(2.0 * PI),
    ) {
        let rotation = Matrix3::from_angle(angle);
        
        // Rotation matrices should preserve length
        let test_vector = Vector2::new(1.0, 0.0);
        let rotated = rotation.transform_point2(test_vector);
        let original_length = test_vector.length();
        let rotated_length = rotated.length();
        
        let epsilon = 0.001;
        prop_assert!((original_length - rotated_length).abs() < epsilon);
        
        // Determinant of rotation matrix should be 1
        let det = rotation.determinant();
        prop_assert!((det - 1.0).abs() < epsilon);
    }
}
