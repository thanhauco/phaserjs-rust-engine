//! Game loop implementation

use std::collections::VecDeque;

/// Game loop state
#[derive(Debug)]
pub struct GameLoopState {
    pub running: bool,
    pub paused: bool,
    pub last_time: f64,
    pub current_time: f64,
    pub delta: f64,
    pub frame_count: u64,
    pub target_fps: Option<f32>,
    fps_history: VecDeque<f64>,
    fps_update_time: f64,
}

impl GameLoopState {
    pub fn new() -> Self {
        Self {
            running: false,
            paused: false,
            last_time: 0.0,
            current_time: 0.0,
            delta: 0.0,
            frame_count: 0,
            target_fps: None,
            fps_history: VecDeque::with_capacity(60),
            fps_update_time: 0.0,
        }
    }
    
    pub fn start(&mut self, time: f64) {
        self.running = true;
        self.last_time = time;
        self.current_time = time;
        self.fps_update_time = time;
    }
    
    pub fn update(&mut self, time: f64) {
        self.current_time = time;
        self.delta = time - self.last_time;
        self.last_time = time;
        self.frame_count += 1;
        
        // Update FPS tracking
        if self.delta > 0.0 {
            self.fps_history.push_back(self.delta);
            if self.fps_history.len() > 60 {
                self.fps_history.pop_front();
            }
        }
    }
    
    pub fn pause(&mut self) {
        self.paused = true;
    }
    
    pub fn resume(&mut self, time: f64) {
        self.paused = false;
        // Reset last_time to avoid large delta after resume
        self.last_time = time;
    }
    
    pub fn stop(&mut self) {
        self.running = false;
    }
    
    pub fn set_target_fps(&mut self, fps: f32) {
        self.target_fps = Some(fps);
    }
    
    pub fn get_fps(&self) -> f32 {
        if self.fps_history.is_empty() {
            return 0.0;
        }
        
        let avg_delta: f64 = self.fps_history.iter().sum::<f64>() / self.fps_history.len() as f64;
        if avg_delta > 0.0 {
            (1000.0 / avg_delta) as f32
        } else {
            0.0
        }
    }
    
    pub fn get_delta_seconds(&self) -> f32 {
        (self.delta / 1000.0) as f32
    }
    
    pub fn is_running(&self) -> bool {
        self.running
    }
    
    pub fn is_paused(&self) -> bool {
        self.paused
    }
}

impl Default for GameLoopState {
    fn default() -> Self {
        Self::new()
    }
}
