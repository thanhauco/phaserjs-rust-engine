//! Phaser Rust Engine - Core Module
//!
//! This module contains the core game engine functionality including:
//! - Game instance and configuration
//! - Game loop management
//! - Event system
//! - Scene management
//! - Canvas management

pub mod canvas;
pub mod config;
pub mod error;
pub mod events;
pub mod game;
pub mod game_loop;

pub use canvas::Canvas;
pub use config::{Color, GameConfig, RendererType};
pub use error::EngineError;
pub use game::Game;

/// Result type for engine operations
pub type Result<T> = std::result::Result<T, EngineError>;
