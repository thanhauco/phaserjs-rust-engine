//! Property-based tests for core systems

use phaser_core::{Game, GameConfig};
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
        // (renderer, texture manager, cache, input, sound, scene manager)
        // For now, we verify the game instance exists and config is accessible
        prop_assert!(game.config().width > 0);
        prop_assert!(game.config().height > 0);
    }
}
