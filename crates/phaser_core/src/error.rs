//! Error types for the Phaser Rust Engine

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    // Initialization errors
    #[error("Renderer initialization failed: {0}")]
    RendererInitFailed(String),
    
    #[error("System initialization failed: {0}")]
    SystemInitFailed(String),
    
    // Asset loading errors
    #[error("Asset not found: {0}")]
    AssetNotFound(String),
    
    #[error("Asset load failed: {path} - {reason}")]
    AssetLoadFailed { path: String, reason: String },
    
    #[error("Invalid asset format: {0}")]
    InvalidAssetFormat(String),
    
    // Scene errors
    #[error("Scene not found: {0}")]
    SceneNotFound(String),
    
    #[error("Scene already exists: {0}")]
    SceneAlreadyExists(String),
    
    #[error("Invalid scene transition from {from} to {to}")]
    InvalidSceneTransition { from: String, to: String },
    
    // GameObject errors
    #[error("Game object not found: {0}")]
    GameObjectNotFound(u64),
    
    #[error("Invalid transform: {0}")]
    InvalidTransform(String),
    
    // Physics errors
    #[error("Invalid collision shape: {0}")]
    InvalidCollisionShape(String),
    
    #[error("Physics world error: {0}")]
    PhysicsWorldError(String),
    
    // AI/ML errors
    #[error("Model load failed: {path} - {reason}")]
    ModelLoadFailed { path: String, reason: String },
    
    #[error("Invalid model format: {0}")]
    InvalidModelFormat(String),
    
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    
    #[error("Invalid tensor shape: expected {expected:?}, got {got:?}")]
    InvalidTensorShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    
    #[error("LLM connection failed: {0}")]
    LLMConnectionFailed(String),
    
    #[error("LLM request failed: {0}")]
    LLMRequestFailed(String),
    
    // IO errors
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
}
