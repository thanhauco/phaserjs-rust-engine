//! Main game instance

use crate::{
    canvas::Canvas,
    config::{GameConfig, RendererType},
    error::EngineError,
    events::EventEmitter,
    game_loop::GameLoopState,
    scene_manager::SceneManager,
    Result,
};

/// Main game instance
pub struct Game {
    config: GameConfig,
    canvas: Canvas,
    events: EventEmitter,
    loop_state: GameLoopState,
    scene_manager: SceneManager,
    renderer_initialized: bool,
    systems_initialized: bool,
}

impl Game {
    /// Create a new game instance with the given configuration
    pub fn new(config: GameConfig) -> Result<Self> {
        // Create canvas with configured dimensions and background color
        let canvas = Canvas::new(config.width, config.height, config.background_color)?;
        
        let mut game = Self {
            config,
            canvas,
            events: EventEmitter::new(),
            loop_state: GameLoopState::new(),
            scene_manager: SceneManager::new(),
            renderer_initialized: false,
            systems_initialized: false,
        };
        
        // Initialize core systems
        game.initialize_systems()?;
        
        Ok(game)
    }
    
    /// Initialize all core systems
    fn initialize_systems(&mut self) -> Result<()> {
        // Initialize renderer based on configuration
        self.initialize_renderer()?;
        
        // Initialize other core systems
        // - Texture manager
        // - Cache
        // - Input
        // - Sound
        // - Scene manager
        self.systems_initialized = true;
        
        Ok(())
    }
    
    /// Initialize the renderer
    fn initialize_renderer(&mut self) -> Result<()> {
        match self.config.renderer_type {
            RendererType::WebGL => {
                // WebGL renderer initialization will go here
                self.renderer_initialized = true;
            }
            RendererType::Canvas => {
                // Canvas renderer initialization will go here
                self.renderer_initialized = true;
            }
        }
        
        Ok(())
    }
    
    /// Check if all systems are initialized
    pub fn is_initialized(&self) -> bool {
        self.renderer_initialized && self.systems_initialized
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
            let delta = self.loop_state.delta;
            // Update scenes
            self.scene_manager.update(time, delta);
        }
        
        // Render scenes
        self.scene_manager.render();
    }
    
    /// Pause the game
    pub fn pause(&mut self) {
        self.loop_state.pause();
    }
    
    /// Resume the game
    pub fn resume(&mut self) {
        let time = Self::get_time();
        self.loop_state.resume(time);
    }
    
    /// Destroy the game and release resources
    pub fn destroy(&mut self) {
        self.loop_state.stop();
        // Cleanup logic will go here
    }
    
    /// Get the current FPS
    pub fn get_fps(&self) -> f32 {
        self.loop_state.get_fps()
    }
    
    /// Get the delta time in seconds
    pub fn get_delta(&self) -> f32 {
        self.loop_state.get_delta_seconds()
    }
    
    /// Check if the game loop is running
    pub fn is_running(&self) -> bool {
        self.loop_state.is_running()
    }
    
    /// Check if the game is paused
    pub fn is_paused(&self) -> bool {
        self.loop_state.is_paused()
    }
    
    /// Set target FPS
    pub fn set_target_fps(&mut self, fps: f32) {
        self.loop_state.set_target_fps(fps);
    }
    
    /// Get frame count
    pub fn get_frame_count(&self) -> u64 {
        self.loop_state.frame_count
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &GameConfig {
        &self.config
    }
    
    /// Get the canvas
    pub fn canvas(&self) -> &Canvas {
        &self.canvas
    }
    
    /// Get mutable canvas
    pub fn canvas_mut(&mut self) -> &mut Canvas {
        &mut self.canvas
    }
    
    /// Get the event emitter
    pub fn events(&mut self) -> &mut EventEmitter {
        &mut self.events
    }
    
    /// Get the renderer type
    pub fn renderer_type(&self) -> RendererType {
        self.config.renderer_type
    }
    
    /// Get the scene manager
    pub fn scene_manager(&self) -> &SceneManager {
        &self.scene_manager
    }
    
    /// Get mutable scene manager
    pub fn scene_manager_mut(&mut self) -> &mut SceneManager {
        &mut self.scene_manager
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
