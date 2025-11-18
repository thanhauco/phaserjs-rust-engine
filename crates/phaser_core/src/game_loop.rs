//! Game loop implementation

/// Game loop state
#[derive(Debug)]
pub struct GameLoopState {
    pub running: bool,
    pub paused: bool,
    pub last_time: f64,
    pub current_time: f64,
    pub delta: f64,
    pub frame_count: u64,
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
        }
    }
    
    pub fn start(&mut self, time: f64) {
        self.running = true;
        self.last_time = time;
        self.current_time = time;
    }
    
    pub fn update(&mut self, time: f64) {
        self.current_time = time;
        self.delta = time - self.last_time;
        self.last_time = time;
        self.frame_count += 1;
    }
    
    pub fn pause(&mut self) {
        self.paused = true;
    }
    
    pub fn resume(&mut self) {
        self.paused = false;
    }
    
    pub fn stop(&mut self) {
        self.running = false;
    }
}

impl Default for GameLoopState {
    fn default() -> Self {
        Self::new()
    }
}
