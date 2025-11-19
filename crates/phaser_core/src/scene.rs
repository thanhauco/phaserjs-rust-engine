//! Scene system for organizing game content

use crate::Result;

/// Scene trait that all scenes must implement
pub trait Scene {
    /// Initialize the scene with optional data
    fn init(&mut self, _data: SceneData) {}
    
    /// Preload assets for this scene
    fn preload(&mut self) {}
    
    /// Create game objects and set up the scene
    fn create(&mut self) {}
    
    /// Update the scene logic
    fn update(&mut self, _time: f64, _delta: f64) {}
    
    /// Render the scene (optional, for custom rendering)
    fn render(&self) {}
    
    /// Called when the scene is being shut down
    fn shutdown(&mut self) {}
    
    /// Called when the scene is being destroyed
    fn destroy(&mut self) {}
    
    /// Get the scene key/name
    fn key(&self) -> &str;
}

/// Data passed to a scene during initialization
#[derive(Debug, Clone, Default)]
pub struct SceneData {
    pub data: std::collections::HashMap<String, String>,
}

impl SceneData {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_data(mut self, key: String, value: String) -> Self {
        self.data.insert(key, value);
        self
    }
    
    pub fn get(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
}

/// Scene lifecycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneState {
    Pending,
    Init,
    Preload,
    Create,
    Running,
    Paused,
    Shutdown,
    Destroyed,
}

/// Internal scene wrapper
pub struct SceneWrapper {
    scene: Box<dyn Scene>,
    state: SceneState,
    active: bool,
    visible: bool,
}

impl SceneWrapper {
    pub fn new(scene: Box<dyn Scene>) -> Self {
        Self {
            scene,
            state: SceneState::Pending,
            active: false,
            visible: true,
        }
    }
    
    pub fn start(&mut self, data: SceneData) -> Result<()> {
        self.state = SceneState::Init;
        self.scene.init(data);
        
        self.state = SceneState::Preload;
        self.scene.preload();
        
        self.state = SceneState::Create;
        self.scene.create();
        
        self.state = SceneState::Running;
        self.active = true;
        
        Ok(())
    }
    
    pub fn update(&mut self, time: f64, delta: f64) {
        if self.active && self.state == SceneState::Running {
            self.scene.update(time, delta);
        }
    }
    
    pub fn render(&self) {
        if self.visible && self.state == SceneState::Running {
            self.scene.render();
        }
    }
    
    pub fn pause(&mut self) {
        if self.state == SceneState::Running {
            self.state = SceneState::Paused;
            self.active = false;
        }
    }
    
    pub fn resume(&mut self) {
        if self.state == SceneState::Paused {
            self.state = SceneState::Running;
            self.active = true;
        }
    }
    
    pub fn shutdown(&mut self) {
        self.state = SceneState::Shutdown;
        self.active = false;
        self.scene.shutdown();
    }
    
    pub fn destroy(&mut self) {
        self.shutdown();
        self.state = SceneState::Destroyed;
        self.scene.destroy();
    }
    
    pub fn key(&self) -> &str {
        self.scene.key()
    }
    
    pub fn state(&self) -> SceneState {
        self.state
    }
    
    pub fn is_active(&self) -> bool {
        self.active
    }
    
    pub fn is_visible(&self) -> bool {
        self.visible
    }
}
