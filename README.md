# Phaser Rust Engine

A high-performance 2D game engine written in Rust, inspired by Phaser 3, with advanced AI/ML capabilities.

## Features

### Core Engine
- WebGL and Canvas 2D rendering
- Scene management with lifecycle methods
- Game object system with transforms
- Asset loading and caching
- Input handling (keyboard, mouse, touch, gamepad)
- Physics systems (Arcade and Matter.js-style)
- Animation and tween systems
- Tilemap support
- Particle systems
- Audio playback
- Camera system with effects

### AI/ML Features
- Neural network inference (ONNX support)
- Reinforcement learning agents
- LLM integration for dynamic dialogue
- Procedural content generation
- Advanced pathfinding with learned heuristics
- Computer vision (object detection, style transfer)
- Behavior trees with ML-enhanced decision making
- NLP (sentiment analysis, speech recognition/synthesis)
- Player behavior prediction
- Anomaly detection for anti-cheat
- Multi-agent coordination

## Project Structure

```
phaser_rust_engine/
├── crates/
│   ├── phaser_core/        # Core game engine
│   ├── phaser_math/        # Math utilities
│   ├── phaser_renderer/    # Rendering systems
│   ├── phaser_gameobjects/ # Game objects
│   ├── phaser_physics/     # Physics engines
│   ├── phaser_input/       # Input handling
│   ├── phaser_loader/      # Asset loading
│   ├── phaser_audio/       # Audio system
│   ├── phaser_animations/  # Animation system
│   ├── phaser_tweens/      # Tween system
│   ├── phaser_tilemaps/    # Tilemap system
│   ├── phaser_particles/   # Particle system
│   ├── phaser_cameras/     # Camera system
│   └── phaser_ai/          # AI/ML systems
└── examples/               # Example games and demos

```

## Getting Started

### Prerequisites

- Rust 1.70 or higher
- Cargo

### Building

```bash
cargo build
```

### Running Tests

```bash
cargo test
```

### Running Examples

```bash
cargo run --example basic_game
```

## Development Status

This project is in active development. See `.kiro/specs/phaser-rust-engine/` for detailed specifications and implementation tasks.

## License

MIT

## Author

Thanh Vu <thanhauco@gmail.com>
