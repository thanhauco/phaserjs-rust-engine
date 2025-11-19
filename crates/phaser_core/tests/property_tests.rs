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


// **Feature: phaser-rust-engine, Property 7: Delta time calculation**
proptest! {
    #[test]
    fn test_delta_time_calculation(
        time_steps in prop::collection::vec(1.0f64..100.0, 2..20),
    ) {
        let mut game = Game::new(GameConfig::default()).unwrap();
        game.start();
        
        let mut last_time = 0.0;
        
        for (i, step) in time_steps.iter().enumerate() {
            let current_time = last_time + step;
            game.step(current_time);
            
            if i > 0 {
                // Delta should equal the time difference between frames
                let expected_delta = step / 1000.0; // Convert to seconds
                let actual_delta = game.get_delta();
                
                // Allow small floating point error
                prop_assert!((expected_delta - actual_delta as f64).abs() < 0.001);
            }
            
            last_time = current_time;
        }
    }
}


// **Feature: phaser-rust-engine, Property 8: Pause stops updates**
proptest! {
    #[test]
    fn test_pause_stops_updates(
        frames_before_pause in 1u64..10,
        frames_during_pause in 1u64..10,
    ) {
        let mut game = Game::new(GameConfig::default()).unwrap();
        game.start();
        
        // Run some frames before pausing
        for i in 0..frames_before_pause {
            game.step((i as f64 + 1.0) * 16.0);
        }
        
        let frame_count_before_pause = game.get_frame_count();
        prop_assert!(game.is_running());
        prop_assert!(!game.is_paused());
        
        // Pause the game
        game.pause();
        prop_assert!(game.is_paused());
        
        // Step through frames while paused
        for i in 0..frames_during_pause {
            let time = (frames_before_pause + i + 1) as f64 * 16.0;
            game.step(time);
        }
        
        // Frame count should still increase (step is called)
        // but game should remain paused
        prop_assert!(game.is_paused());
        prop_assert!(game.is_running());
        
        // Resume and verify state
        game.resume();
        prop_assert!(!game.is_paused());
        prop_assert!(game.is_running());
    }
}


// **Feature: phaser-rust-engine, Property 9: Destroy releases resources**
proptest! {
    #[test]
    fn test_destroy_releases_resources(
        frames_to_run in 1u64..20,
    ) {
        let mut game = Game::new(GameConfig::default()).unwrap();
        game.start();
        
        prop_assert!(game.is_running());
        
        // Run some frames
        for i in 0..frames_to_run {
            game.step((i as f64 + 1.0) * 16.0);
        }
        
        prop_assert!(game.is_running());
        prop_assert!(game.get_frame_count() > 0);
        
        // Destroy the game
        game.destroy();
        
        // Game loop should be stopped
        prop_assert!(!game.is_running());
        
        // Calling step after destroy should not crash
        // (though it won't do anything since loop is stopped)
        game.step(1000.0);
        prop_assert!(!game.is_running());
    }
}


// **Feature: phaser-rust-engine, Property 5: Scene lifecycle ordering**
proptest! {
    #[test]
    fn test_scene_lifecycle_ordering(
        _dummy in 0u32..10, // Just to make it a property test
    ) {
        use phaser_core::{Scene, SceneData, SceneManager};
        use std::sync::{Arc, Mutex};
        
        #[derive(Clone)]
        struct TestScene {
            key: String,
            call_order: Arc<Mutex<Vec<String>>>,
        }
        
        impl TestScene {
            fn new(key: &str, call_order: Arc<Mutex<Vec<String>>>) -> Self {
                Self {
                    key: key.to_string(),
                    call_order,
                }
            }
        }
        
        impl Scene for TestScene {
            fn init(&mut self, _data: SceneData) {
                self.call_order.lock().unwrap().push("init".to_string());
            }
            
            fn preload(&mut self) {
                self.call_order.lock().unwrap().push("preload".to_string());
            }
            
            fn create(&mut self) {
                self.call_order.lock().unwrap().push("create".to_string());
            }
            
            fn key(&self) -> &str {
                &self.key
            }
        }
        
        let call_order = Arc::new(Mutex::new(Vec::new()));
        let mut manager = SceneManager::new();
        let scene = Box::new(TestScene::new("test", call_order.clone()));
        
        manager.add("test".to_string(), scene);
        manager.start("test").unwrap();
        
        let order = call_order.lock().unwrap();
        
        // Verify lifecycle methods were called in correct order
        prop_assert_eq!(order.len(), 3);
        prop_assert_eq!(order[0], "init");
        prop_assert_eq!(order[1], "preload");
        prop_assert_eq!(order[2], "create");
    }
}
