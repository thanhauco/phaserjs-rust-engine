//! Canvas management

use crate::{config::Color, Result};

/// Canvas representation
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    pub background_color: Color,
}

impl Canvas {
    /// Create a new canvas with the given dimensions
    pub fn new(width: u32, height: u32, background_color: Color) -> Result<Self> {
        Ok(Self {
            width,
            height,
            background_color,
        })
    }
    
    /// Get the canvas width
    pub fn width(&self) -> u32 {
        self.width
    }
    
    /// Get the canvas height
    pub fn height(&self) -> u32 {
        self.height
    }
    
    /// Set the canvas dimensions
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
    
    /// Get the background color
    pub fn background_color(&self) -> Color {
        self.background_color
    }
    
    /// Set the background color
    pub fn set_background_color(&mut self, color: Color) {
        self.background_color = color;
    }
    
    /// Get the aspect ratio
    pub fn aspect_ratio(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canvas_creation() {
        let canvas = Canvas::new(800, 600, Color::BLACK).unwrap();
        assert_eq!(canvas.width(), 800);
        assert_eq!(canvas.height(), 600);
        assert_eq!(canvas.background_color(), Color::BLACK);
    }

    #[test]
    fn test_canvas_resize() {
        let mut canvas = Canvas::new(800, 600, Color::BLACK).unwrap();
        canvas.set_size(1024, 768);
        assert_eq!(canvas.width(), 1024);
        assert_eq!(canvas.height(), 768);
    }

    #[test]
    fn test_aspect_ratio() {
        let canvas = Canvas::new(800, 600, Color::BLACK).unwrap();
        assert!((canvas.aspect_ratio() - 4.0 / 3.0).abs() < 0.001);
    }
}
