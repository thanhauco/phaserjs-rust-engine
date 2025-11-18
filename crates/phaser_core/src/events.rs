//! Event system for the game engine

use std::collections::HashMap;

/// Event emitter for game events
pub struct EventEmitter {
    listeners: HashMap<String, Vec<Box<dyn Fn(&Event) + Send + Sync>>>,
}

impl EventEmitter {
    pub fn new() -> Self {
        Self {
            listeners: HashMap::new(),
        }
    }
    
    pub fn on<F>(&mut self, event_name: &str, callback: F)
    where
        F: Fn(&Event) + Send + Sync + 'static,
    {
        self.listeners
            .entry(event_name.to_string())
            .or_insert_with(Vec::new)
            .push(Box::new(callback));
    }
    
    pub fn emit(&self, event_name: &str, event: &Event) {
        if let Some(callbacks) = self.listeners.get(event_name) {
            for callback in callbacks {
                callback(event);
            }
        }
    }
}

impl Default for EventEmitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Event data
#[derive(Debug, Clone)]
pub struct Event {
    pub name: String,
    pub data: EventData,
}

/// Event data variants
#[derive(Debug, Clone)]
pub enum EventData {
    None,
    String(String),
    Number(f64),
    Bool(bool),
}
