# 4x4 Tic-Tac-Toe with Depth-Limited Minimax Algorithm

## Project Overview

This project implements a 4x4 Tic-Tac-Toe game using a depth-limited minimax algorithm with alpha-beta pruning. The AI opponent has three difficulty levels (easy, medium, hard) controlled by different search depths and evaluation functions.

## Features

- **4x4 Game Board**: Extended version of classic Tic-Tac-Toe
- **Depth-Limited Minimax**: AI uses minimax algorithm with configurable depth limits
- **Alpha-Beta Pruning**: Optional optimization to reduce nodes evaluated
- **Three Difficulty Levels**:
  - **Easy**: Depth = 1, Simple mark count evaluation with random noise
  - **Medium**: Depth = 2, Line potential evaluation with occasional suboptimal moves
  - **Hard**: Depth = 4, Advanced evaluation with center control
- **Comprehensive Statistics**: Track performance metrics for each difficulty level
- **Modern GUI**: Dark-themed interface with real-time statistics and comparisons

## Requirements

- Python 3.6+
- tkinter (usually included with Python)

## Installation

No additional packages required! Just run:

```bash
python tic_tac_toe.py
```

## How to Play

1. **Select Difficulty**: Choose Easy, Medium, or Hard
2. **Toggle Pruning**: Enable/disable alpha-beta pruning to compare performance
3. **Make Moves**: Click on empty cells to place your 'O'
4. **Watch Statistics**: Monitor nodes evaluated, move times, and game outcomes
5. **Compare**: Use the statistics panel to compare different difficulty levels

## Game Rules

- You play as 'O' (green)
- AI plays as 'X' (red)
- First to get 4 in a row (horizontal, vertical, or diagonal) wins
- If the board fills up with no winner, it's a draw

## Algorithm Details

### Minimax Algorithm
The minimax algorithm evaluates all possible moves up to a certain depth, choosing moves that maximize the AI's score while minimizing the human player's score.

### Evaluation Functions

Each difficulty level uses a different evaluation function:

#### Easy Level (Depth = 1)
- **Terminal States**: +10000 (AI win), -10000 (Human win), 0 (draw)
- **Non-terminal**: Simple mark count difference (AI marks - Human marks)
- **Random Noise**: Adds random.uniform(-0.5, 0.5) to introduce variability
- **Strategy**: Very weak heuristic that makes the AI easier to beat

#### Medium Level (Depth = 2)
- **Terminal States**: +10000 (AI win), -10000 (Human win), 0 (draw)
- **Non-terminal**: Line-based potential analysis
  - 1 mark in a line: ±1
  - 2 marks in a line: ±4
  - 3 marks in a line: ±0 (ignored - no bonus/penalty)
- **Move Selection**: 25% chance to choose 2nd best move instead of best move
- **Strategy**: Focuses on building 2-in-a-row positions while ignoring immediate threats

#### Hard Level (Depth = 4)
- **Terminal States**: +100000 (AI win), -100000 (Human win), 0 (draw)
- **Non-terminal**: Advanced line analysis + center control
  - 1 mark in a line: ±2
  - 2 marks in a line: ±8
  - 3 marks in a line: ±50 (strong threat/defense)
  - Center positions bonus: ±3 per center cell
- **Strategy**: Strong strategic play with deep lookahead and positional awareness

### Alpha-Beta Pruning
Alpha-beta pruning optimizes the minimax algorithm by eliminating branches that cannot affect the final decision, significantly reducing the number of nodes evaluated without changing the result.

## Statistics Tracked

### Current Game
- Nodes evaluated (cumulative)
- Total move time (cumulative)
- Moves made
- Current depth and difficulty

### Per Difficulty Level
- AI Wins / Human Wins / Draws
- **Win Rate Percentage** (automatically calculated)
- Average nodes evaluated per move
- Average move time
- Total moves made

### Pruning Comparison
- Visual indicator of pruning status
- Compare performance with and without pruning

## Performance Comparison

The statistics panel allows you to compare:
- **Different Depths**: See how depth affects decision quality and computation time
- **Different Evaluation Functions**: Observe how evaluation functions affect AI behavior
- **With/Without Pruning**: Observe the performance improvement from alpha-beta pruning
- **Difficulty Levels**: Track win rates and performance metrics across difficulties

## GUI Features

### Main Game Area
- 4x4 interactive game board
- Color-coded moves (X=red, O=green)
- Real-time game status display

### Control Panel
- Difficulty selection (Easy/Medium/Hard)
- Alpha-beta pruning toggle
- New Game button
- Reset Statistics button

### Statistics Panel
- **Current Game Stats**: Real-time metrics for the active game
- **Difficulty Comparison Tabs**: Separate tabs showing statistics for each difficulty
- **Evaluation Function Info**: Displays which evaluation function is active
- **Pruning Comparison**: Shows pruning status and benefits
- **Depth Information**: Shows depth settings for each difficulty

## Reflection Questions

1. **How does depth limit affect AI's ability?**
   - Deeper searches allow the AI to see further ahead, making more strategic decisions
   - However, deeper searches require more computation time
   - Easy (depth 1) makes quick decisions but misses long-term strategy
   - Hard (depth 4) sees much further ahead but takes longer to compute

2. **How does depth affect computational cost?**
   - Computational cost grows exponentially with depth
   - Depth 4 evaluates significantly more nodes than depth 1
   - Alpha-beta pruning can dramatically reduce nodes evaluated
   - Medium level evaluates all moves (for move selection), which increases computation

3. **How do evaluation functions affect gameplay?**
   - Easy's simple mark count makes the AI weak and unpredictable
   - Medium's line-based approach focuses on 2-in-a-row positions
   - Hard's advanced evaluation considers threats, defense, and center control
   - Different evaluation functions create distinct playing styles

4. **In which scenarios might depth-limited minimax be suboptimal?**
   - Even at depth 4, the AI might miss very long-term strategies
   - In complex positions, deeper search might reveal better moves
   - Terminal states beyond the depth limit are evaluated heuristically, not perfectly
   - Medium level's random move selection can lead to suboptimal play

5. **How does alpha-beta pruning affect performance?**
   - Pruning reduces nodes evaluated without changing the result
   - The improvement is more significant at deeper search depths
   - Without pruning, the AI evaluates many unnecessary branches
   - With pruning, the AI can search deeper in the same time

## Project Structure

```
Ai-Project2/
├── tic_tac_toe.py    # Main game implementation
└── README.md         # This file
```

## Key Implementation Details

### Evaluation Function Differences
- **Easy**: Uses simple mark counting with random noise - makes the AI weak and unpredictable
- **Medium**: Line-based evaluation that ignores 3-in-a-row threats - focuses on building positions
- **Hard**: Advanced evaluation with center control - considers multiple strategic factors

### Move Selection
- **Easy/Hard**: Standard minimax - always chooses the best move
- **Medium**: Evaluates all moves, then randomly chooses between best and 2nd best (25% chance for 2nd best)

### Terminal State Handling
- Easy and Medium use ±10000 for terminal states
- Hard uses ±100000 for terminal states (larger magnitude to ensure proper prioritization)

## Author

USJ-ESIB Artificial Intelligence Project

## Notes

- The game tracks statistics separately for each difficulty level
- Win rates are automatically calculated and displayed
- You can reset statistics at any time
- The GUI updates in real-time as you play
- Compare different difficulty levels and pruning settings to observe the differences
