//! Main game instance

use crate::{config::GameConfig, error::EngineError, events::EventEmitter, game_loop::GameLoopState, Result};

/// Main game instance
pub struct Game {
    config: GameConfig,
    events: EventEmitter,
    loop_state: GameLoopState,
}

impl Game {
    /// Create a new game instance with the given configuration
    pub fn new(config: GameConfig) -> Result<Self> {
        Ok(Self {
            config,
            events: EventEmitter::new(),
            loop_state: GameLoopState::new(),
        })
    }
    
    /// Start the game loop
    pub fn start(&mut self) {
        let time = Self::get_time();
        self.loop_state.start(time);
    }
    
    /// Execute a single frame step
    pub fn step(&mut self, time: f64) {
        self.loop_state.update(time);
        
        if !self.loop_state.paused {
            // Update logic will go here
        }
        
        // Render logic will go here
    }
    
    /// Pause the game
    pub fn pause(&mut self) {
        self.loop_state.pause();
    }
    
    /// Resume the game
    pub fn resume(&mut self) {
        self.loop_state.resume();
    }
    
    /// Destroy the game and release resources
    pub fn destroy(&mut self) {
        self.loop_state.stop();
        // Cleanup logic will go here
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &GameConfig {
        &self.config
    }
    
    /// Get the event emitter
    pub fn events(&mut self) -> &mut EventEmitter {
        &mut self.events
    }
    
    /// Get current time in milliseconds
    fn get_time() -> f64 {
        // Platform-specific time implementation
        #[cfg(target_arch = "wasm32")]
        {
            web_sys::window()
                .expect("no window")
                .performance()
                .expect("no performance")
                .now()
        }
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_secs_f64()
                * 1000.0
        }
    }
}
