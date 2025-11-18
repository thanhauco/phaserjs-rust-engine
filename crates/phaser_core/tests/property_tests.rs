//! Property-based tests for core systems

use phaser_core::{Game, GameConfig, RendererType};
use proptest::prelude::*;

// **Feature: phaser-rust-engine, Property 3: Core systems initialization completeness**
proptest! {
    #[test]
    fn test_core_systems_initialization(
        width in 100u32..2000u32,
        height in 100u32..2000u32,
    ) {
        let mut config = GameConfig::default();
        config.width = width;
        config.height = height;
        
        let game = Game::new(config);
        
        // Verify game instance was created successfully
        prop_assert!(game.is_ok());
        
        let game = game.unwrap();
        
        // Verify configuration was stored correctly
        prop_assert_eq!(game.config().width, width);
        prop_assert_eq!(game.config().height, height);
        
        // All core systems should be initialized
        prop_assert!(game.is_initialized());
        
        // Canvas should be created with correct dimensions
        prop_assert_eq!(game.canvas().width(), width);
        prop_assert_eq!(game.canvas().height(), height);
    }
}

// **Feature: phaser-rust-engine, Property 1: Renderer initialization matches configuration**
proptest! {
    #[test]
    fn test_renderer_initialization_matches_config(
        width in 100u32..2000u32,
        height in 100u32..2000u32,
        use_webgl in prop::bool::ANY,
    ) {
        let mut config = GameConfig::default();
        config.width = width;
        config.height = height;
        config.renderer_type = if use_webgl {
            RendererType::WebGL
        } else {
            RendererType::Canvas
        };
        
        let game = Game::new(config.clone());
        
        prop_assert!(game.is_ok());
        
        let game = game.unwrap();
        
        // Verify renderer type matches configuration
        prop_assert_eq!(game.renderer_type(), config.renderer_type);
        
        // Verify renderer was initialized
        prop_assert!(game.is_initialized());
    }
}

// **Feature: phaser-rust-engine, Property 2: Canvas dimensions match configuration**
proptest! {
    #[test]
    fn test_canvas_dimensions_match_config(
        width in 100u32..4000u32,
        height in 100u32..4000u32,
    ) {
        let mut config = GameConfig::default();
        config.width = width;
        config.height = height;
        
        let game = Game::new(config);
        
        prop_assert!(game.is_ok());
        
        let game = game.unwrap();
        
        // Canvas dimensions should exactly match configuration
        prop_assert_eq!(game.canvas().width(), width);
        prop_assert_eq!(game.canvas().height(), height);
        
        // Aspect ratio should be correct
        let expected_aspect = width as f32 / height as f32;
        let actual_aspect = game.canvas().aspect_ratio();
        prop_assert!((expected_aspect - actual_aspect).abs() < 0.001);
    }
}

// **Feature: phaser-rust-engine, Property 4: Background color application**
proptest! {
    #[test]
    fn test_background_color_application(
        r in 0u8..=255,
        g in 0u8..=255,
        b in 0u8..=255,
        a in 0u8..=255,
    ) {
        use phaser_core::Color;
        
        let mut config = GameConfig::default();
        let color = Color::new(r, g, b, a);
        config.background_color = color;
        
        let game = Game::new(config);
        
        prop_assert!(game.is_ok());
        
        let game = game.unwrap();
        
        // Background color should be applied immediately to canvas
        let canvas_color = game.canvas().background_color();
        prop_assert_eq!(canvas_color.r, r);
        prop_assert_eq!(canvas_color.g, g);
        prop_assert_eq!(canvas_color.b, b);
        prop_assert_eq!(canvas_color.a, a);
    }
}
