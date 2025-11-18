//! Game configuration types

use serde::{Deserialize, Serialize};

/// Main game configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameConfig {
    /// Canvas width in pixels
    pub width: u32,
    /// Canvas height in pixels
    pub height: u32,
    /// Renderer type to use
    pub renderer_type: RendererType,
    /// Parent container element ID (for web)
    pub parent: Option<String>,
    /// Background color
    pub background_color: Color,
    /// Scale configuration
    pub scale: ScaleConfig,
    /// Physics configuration
    pub physics: PhysicsConfig,
    /// Audio configuration
    pub audio: AudioConfig,
    /// AI/ML configuration
    pub ai: AIConfig,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            renderer_type: RendererType::WebGL,
            parent: None,
            background_color: Color::BLACK,
            scale: ScaleConfig::default(),
            physics: PhysicsConfig::default(),
            audio: AudioConfig::default(),
            ai: AIConfig::default(),
        }
    }
}

/// Renderer type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RendererType {
    WebGL,
    Canvas,
}

/// Color representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    pub const BLACK: Color = Color { r: 0, g: 0, b: 0, a: 255 };
    pub const WHITE: Color = Color { r: 255, g: 255, b: 255, a: 255 };
    
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
    
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }
}

/// Scale configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleConfig {
    pub mode: ScaleMode,
    pub auto_center: bool,
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self {
            mode: ScaleMode::None,
            auto_center: false,
        }
    }
}

/// Scale mode options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleMode {
    None,
    Fit,
    Envelop,
    Resize,
}

/// Physics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    pub default_system: PhysicsSystem,
    pub arcade: ArcadePhysicsConfig,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            default_system: PhysicsSystem::Arcade,
            arcade: ArcadePhysicsConfig::default(),
        }
    }
}

/// Physics system type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsSystem {
    Arcade,
    Matter,
}

/// Arcade physics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcadePhysicsConfig {
    pub gravity_x: f32,
    pub gravity_y: f32,
    pub debug: bool,
}

impl Default for ArcadePhysicsConfig {
    fn default() -> Self {
        Self {
            gravity_x: 0.0,
            gravity_y: 0.0,
            debug: false,
        }
    }
}

/// Audio configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub disable_web_audio: bool,
    pub no_audio: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            disable_web_audio: false,
            no_audio: false,
        }
    }
}

/// AI/ML configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIConfig {
    pub enable_inference: bool,
    pub inference_backend: InferenceBackend,
    pub enable_llm: bool,
}

impl Default for AIConfig {
    fn default() -> Self {
        Self {
            enable_inference: false,
            inference_backend: InferenceBackend::CPU,
            enable_llm: false,
        }
    }
}

/// Inference backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceBackend {
    CPU,
    GPU,
    WASM,
}
