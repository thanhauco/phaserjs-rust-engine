//! Scene manager for handling multiple scenes

use crate::{
    scene::{Scene, SceneData, SceneState, SceneWrapper},
    Result,
};
use std::collections::HashMap;

/// Scene manager handles scene lifecycle and transitions
pub struct SceneManager {
    scenes: HashMap<String, SceneWrapper>,
    active_scenes: Vec<String>,
}

impl SceneManager {
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
            active_scenes: Vec::new(),
        }
    }
    
    /// Add a scene to the manager
    pub fn add(&mut self, key: String, scene: Box<dyn Scene>) {
        let wrapper = SceneWrapper::new(scene);
        self.scenes.insert(key, wrapper);
    }
    
    /// Start a scene
    pub fn start(&mut self, key: &str) -> Result<()> {
        self.start_with_data(key, SceneData::new())
    }
    
    /// Start a scene with initialization data
    pub fn start_with_data(&mut self, key: &str, data: SceneData) -> Result<()> {
        if let Some(scene) = self.scenes.get_mut(key) {
            scene.start(data)?;
            
            if !self.active_scenes.contains(&key.to_string()) {
                self.active_scenes.push(key.to_string());
            }
            
            Ok(())
        } else {
            Err(crate::EngineError::SceneNotFound(key.to_string()))
        }
    }
    
    /// Stop a scene
    pub fn stop(&mut self, key: &str) -> Result<()> {
        if let Some(scene) = self.scenes.get_mut(key) {
            scene.shutdown();
            self.active_scenes.retain(|k| k != key);
            Ok(())
        } else {
            Err(crate::EngineError::SceneNotFound(key.to_string()))
        }
    }
    
    /// Switch from one scene to another
    pub fn switch(&mut self, from: &str, to: &str) -> Result<()> {
        // Shutdown the current scene
        if let Some(scene) = self.scenes.get_mut(from) {
            scene.shutdown();
            self.active_scenes.retain(|k| k != from);
        }
        
        // Start the new scene
        self.start(to)
    }
    
    /// Pause a scene
    pub fn pause(&mut self, key: &str) -> Result<()> {
        if let Some(scene) = self.scenes.get_mut(key) {
            scene.pause();
            Ok(())
        } else {
            Err(crate::EngineError::SceneNotFound(key.to_string()))
        }
    }
    
    /// Resume a scene
    pub fn resume(&mut self, key: &str) -> Result<()> {
        if let Some(scene) = self.scenes.get_mut(key) {
            scene.resume();
            Ok(())
        } else {
            Err(crate::EngineError::SceneNotFound(key.to_string()))
        }
    }
    
    /// Update all active scenes
    pub fn update(&mut self, time: f64, delta: f64) {
        // Update scenes in the order they were added
        for key in &self.active_scenes.clone() {
            if let Some(scene) = self.scenes.get_mut(key) {
                scene.update(time, delta);
            }
        }
    }
    
    /// Render all active scenes
    pub fn render(&self) {
        // Render scenes in the order they were added
        for key in &self.active_scenes {
            if let Some(scene) = self.scenes.get(key) {
                scene.render();
            }
        }
    }
    
    /// Get a scene by key
    pub fn get(&self, key: &str) -> Option<&SceneWrapper> {
        self.scenes.get(key)
    }
    
    /// Get a mutable scene by key
    pub fn get_mut(&mut self, key: &str) -> Option<&mut SceneWrapper> {
        self.scenes.get_mut(key)
    }
    
    /// Check if a scene exists
    pub fn has(&self, key: &str) -> bool {
        self.scenes.contains_key(key)
    }
    
    /// Get the list of active scene keys
    pub fn active_scenes(&self) -> &[String] {
        &self.active_scenes
    }
    
    /// Remove a scene
    pub fn remove(&mut self, key: &str) -> Result<()> {
        if let Some(mut scene) = self.scenes.remove(key) {
            scene.destroy();
            self.active_scenes.retain(|k| k != key);
            Ok(())
        } else {
            Err(crate::EngineError::SceneNotFound(key.to_string()))
        }
    }
}

impl Default for SceneManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct TestScene {
        key: String,
        init_called: bool,
        preload_called: bool,
        create_called: bool,
        update_count: u32,
    }
    
    impl TestScene {
        fn new(key: &str) -> Self {
            Self {
                key: key.to_string(),
                init_called: false,
                preload_called: false,
                create_called: false,
                update_count: 0,
            }
        }
    }
    
    impl Scene for TestScene {
        fn init(&mut self, _data: SceneData) {
            self.init_called = true;
        }
        
        fn preload(&mut self) {
            self.preload_called = true;
        }
        
        fn create(&mut self) {
            self.create_called = true;
        }
        
        fn update(&mut self, _time: f64, _delta: f64) {
            self.update_count += 1;
        }
        
        fn key(&self) -> &str {
            &self.key
        }
    }
    
    #[test]
    fn test_scene_lifecycle() {
        let mut manager = SceneManager::new();
        let scene = Box::new(TestScene::new("test"));
        
        manager.add("test".to_string(), scene);
        manager.start("test").unwrap();
        
        // Scene should be active
        assert!(manager.active_scenes().contains(&"test".to_string()));
        
        // Update the scene
        manager.update(0.0, 16.0);
        
        // Stop the scene
        manager.stop("test").unwrap();
        assert!(!manager.active_scenes().contains(&"test".to_string()));
    }
}
