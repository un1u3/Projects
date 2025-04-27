import chess
import chess.pgn
import chess.engine
import argparse
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class MoveAnalysis:
    """Class to store analysis data for a single move."""
    move_number: int
    player_color: chess.Color
    move_played: chess.Move
    san_move: str
    best_move: Optional[chess.Move]
    best_move_san: Optional[str]
    position_eval_before: float  # in centipawns, positive means white advantage
    position_eval_after: float
    eval_difference: float
    mistake_type: Optional[str] = None  # None, "Inaccuracy", "Mistake", or "Blunder"


class ChessPGNAnalyzer:
    """Analyzer for chess PGN files using Stockfish."""

    def __init__(self, stockfish_path: str = None, depth: int = 18, 
                 inaccuracy_threshold: int = 50, mistake_threshold: int = 150, 
                 blunder_threshold: int = 300, time_limit: float = None):
        """Initialize the analyzer with Stockfish settings.
        
        Args:
            stockfish_path: Path to Stockfish executable
            depth: Analysis depth for Stockfish
            inaccuracy_threshold: CP threshold for inaccuracies (default: 50)
            mistake_threshold: CP threshold for mistakes (default: 150)
            blunder_threshold: CP threshold for blunders (default: 300)
            time_limit: Time limit for analysis in seconds (overrides depth if provided)
        """
        # Set up logger first to avoid the AttributeError
        self.logger = self._setup_logger()
        
        # Then initialize the rest of the attributes
        self.depth = depth
        self.time_limit = time_limit
        self.inaccuracy_threshold = inaccuracy_threshold
        self.mistake_threshold = mistake_threshold
        self.blunder_threshold = blunder_threshold
        self.engine = None
        
        # Find stockfish path last (now can use logger safely)
        self.stockfish_path = stockfish_path or self._find_stockfish()
        
    def _setup_logger(self):
        """Set up a basic logger."""
        logger = logging.getLogger("chess_analyzer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _find_stockfish(self):
        """Try to find Stockfish in common locations."""
        common_paths = [
            "stockfish",  # If in PATH
            "/usr/local/bin/stockfish",
            "/usr/games/stockfish",
            "/usr/bin/stockfish",  # Common location on Linux
            "C:/Program Files/stockfish/stockfish.exe",
            os.path.expanduser("~/stockfish/stockfish")
        ]
        
        for path in common_paths:
            try:
                # Try to run stockfish with a timeout to avoid hanging
                import subprocess
                subprocess.run([path], input=b"quit\n", timeout=1, capture_output=True)
                self.logger.info(f"Found Stockfish at {path}")
                return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
            except Exception as e:
                self.logger.debug(f"Error checking {path}: {e}")
                continue
        
        # If we get here, we couldn't find Stockfish
        self.logger.warning("Stockfish not found in common locations. Please specify path manually.")
        return None

    def start_engine(self):
        """Start the Stockfish engine."""
        if not self.stockfish_path:
            raise ValueError("Stockfish path not set or could not be found automatically. Please specify the path using --stockfish parameter.")
            
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self.logger.info(f"Stockfish engine started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start Stockfish: {e}")
            raise

    def stop_engine(self):
        """Stop the Stockfish engine properly."""
        if self.engine:
            self.engine.quit()
            self.engine = None
            self.logger.info("Stockfish engine stopped")

    def _get_position_evaluation(self, board: chess.Board) -> float:
        """Get position evaluation in centipawns.
        
        Args:
            board: Current chess position
            
        Returns:
            Evaluation in centipawns (positive for white advantage)
        """
        if self.engine is None:
            self.start_engine()
            
        # Set up analysis parameters
        limit_dict = {}
        if self.time_limit:
            limit_dict["time"] = self.time_limit
        else:
            limit_dict["depth"] = self.depth
            
        # Get engine evaluation
        info = self.engine.analyse(board, chess.engine.Limit(**limit_dict))
        
        # Extract score from analysis
        score = info["score"].relative
        
        # Handle mate scores by converting to a high centipawn value
        if score.is_mate():
            mate_score = score.mate()
            # Convert mate score to centipawns (the sign indicates which side has mate)
            return 10000 * (1 if mate_score > 0 else -1)
        
        # Return score in centipawns
        return score.score() or 0

    def _get_best_move(self, board: chess.Board) -> Tuple[Optional[chess.Move], Optional[str]]:
        """Find the best move for the current position.
        
        Args:
            board: Current chess position
            
        Returns:
            Tuple of (best move object, best move in SAN notation)
        """
        if self.engine is None:
            self.start_engine()
            
        # Set up analysis parameters
        limit_dict = {}
        if self.time_limit:
            limit_dict["time"] = self.time_limit
        else:
            limit_dict["depth"] = self.depth
            
        # Get engine's best move
        result = self.engine.play(board, chess.engine.Limit(**limit_dict))
        
        if result.move:
            # Convert to SAN notation
            san_move = board.san(result.move)
            return result.move, san_move
        
        return None, None

    def _classify_mistake(self, eval_diff: float) -> Optional[str]:
        """Classify a move as an inaccuracy, mistake, or blunder based on eval difference.
        
        Args:
            eval_diff: Evaluation difference in centipawns
            
        Returns:
            Classification as string or None if not a mistake
        """
        # Use absolute value for classification
        eval_diff = abs(eval_diff)
        
        if eval_diff >= self.blunder_threshold:
            return "Blunder"
        elif eval_diff >= self.mistake_threshold:
            return "Mistake"
        elif eval_diff >= self.inaccuracy_threshold:
            return "Inaccuracy"
        return None

    def analyze_game(self, game: chess.pgn.Game) -> List[MoveAnalysis]:
        """Analyze a chess game move by move.
        
        Args:
            game: A parsed chess.pgn.Game object
            
        Returns:
            List of MoveAnalysis objects for each move
        """
        self.logger.info(f"Analyzing game: {game.headers.get('White', 'Unknown')} vs {game.headers.get('Black', 'Unknown')}")
        
        if not self.engine:
            self.start_engine()
        
        results = []
        board = game.board()
        node = game
        
        # Track move numbers
        move_number = 0
        player_color = chess.WHITE
        
        while node.variations:
            # Get the mainline move
            node = node.variations[0]
            move = node.move
            
            # Update move tracking (move_number is the full-move number)
            if player_color == chess.WHITE:
                move_number += 1
                
            # Convert move to SAN notation for readability
            san_move = board.san(move)
            
            # Evaluate position before move
            eval_before = self._get_position_evaluation(board)
            
            # Find best move in current position
            best_move, best_move_san = self._get_best_move(board)
            
            # Make the move on our board
            board.push(move)
            
            # Evaluate position after move
            eval_after = -self._get_position_evaluation(board)  # Negate because it's opponent's perspective
            
            # Calculate evaluation difference
            # Positive difference means the move was worse than optimal
            # For black, we need to invert the evaluation to keep consistent (pos = player advantage)
            eval_diff = eval_after - eval_before
            if player_color == chess.BLACK:
                eval_diff = -eval_diff
                
            # Classify the move
            mistake_type = self._classify_mistake(eval_diff)
            
            # Create analysis object
            analysis = MoveAnalysis(
                move_number=move_number,
                player_color=player_color,
                move_played=move,
                san_move=san_move,
                best_move=best_move,
                best_move_san=best_move_san,
                position_eval_before=eval_before,
                position_eval_after=eval_after,
                eval_difference=eval_diff,
                mistake_type=mistake_type
            )
            
            results.append(analysis)
            
            # Toggle player color for next move
            player_color = not player_color
            
        return results

    def analyze_pgn_file(self, pgn_path: str, include_all_moves: bool = False) -> Dict[str, Dict]:
        """Analyze a PGN file containing one or multiple games.
        
        Args:
            pgn_path: Path to the PGN file
            include_all_moves: Whether to include moves without mistakes in results
            
        Returns:
            Dictionary mapping game IDs to dictionaries with headers and analysis results
        """
        # Start the engine if not already running
        if not self.engine:
            self.start_engine()
            
        # Open the PGN file
        try:
            with open(pgn_path, encoding='utf-8-sig') as pgn_file:
                games_analysis = {}
                game_count = 0
                
                # Read each game in the PGN file
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:  # End of file
                        break
                        
                    game_count += 1
                    game_id = self._get_game_identifier(game, game_count)
                    
                    # Analyze the game
                    analysis_results = self.analyze_game(game)
                    
                    # Filter results if needed
                    if not include_all_moves:
                        analysis_results = [a for a in analysis_results if a.mistake_type is not None]
                        
                    games_analysis[game_id] = {
                        'headers': game.headers,
                        'analysis': analysis_results
                    }
                    
                self.logger.info(f"Analyzed {game_count} games from {pgn_path}")
                return games_analysis
                
        except FileNotFoundError:
            self.logger.error(f"PGN file not found: {pgn_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error analyzing PGN file: {e}")
            raise
        finally:
            # Always clean up the engine
            self.stop_engine()

    def _get_game_identifier(self, game: chess.pgn.Game, game_count: int) -> str:
        """Generate a readable identifier for a game.
        
        Args:
            game: Chess game object
            game_count: Number of the game in the PGN file
            
        Returns:
            String identifier
        """
        white = game.headers.get('White', 'Unknown')
        black = game.headers.get('Black', 'Unknown')
        event = game.headers.get('Event', '')
        date = game.headers.get('Date', '')
        
        return f"Game {game_count}: {white} vs {black} ({event} {date})".strip()

    def print_analysis_report(self, games_analysis: Dict[str, Dict], output_file: str = None):
        """Print analysis results in human-readable format.
        
        Args:
            games_analysis: Dictionary of game analyses
            output_file: Optional file path to write report to
        """
        output_lines = []
        
        for game_id, game_data in games_analysis.items():
            headers = game_data['headers']
            analysis = game_data['analysis']
            
            output_lines.append(f"=== {game_id} ===")
            output_lines.append(f"Event: {headers.get('Event', 'Unknown')}")
            output_lines.append(f"Date: {headers.get('Date', 'Unknown')}")
            output_lines.append(f"White: {headers.get('White', 'Unknown')}")
            output_lines.append(f"Black: {headers.get('Black', 'Unknown')}")
            output_lines.append(f"Result: {headers.get('Result', 'Unknown')}")
            output_lines.append("="*60)
            
            if not analysis:
                output_lines.append("No mistakes found in this game.")
                output_lines.append("")
                continue
                
            for move in analysis:
                if move.mistake_type:
                    # Format move notation (e.g. "15. Nf3" or "15... Bc5")
                    color_indicator = "." if move.player_color == chess.WHITE else "..."
                    move_notation = f"{move.move_number}{color_indicator} {move.san_move}"
                    
                    output_lines.append(f"Move {move_notation} ({move.mistake_type})")
                    output_lines.append(f"    - Better move: {move.best_move_san}")
                    output_lines.append(f"    - Eval dropped by {abs(move.eval_difference):.1f} centipawns")
                    output_lines.append("")
            
            output_lines.append("-"*60)
            output_lines.append("")
            
        # Print to console or file
        report_text = "\n".join(output_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"Analysis report written to {output_file}")
        else:
            print(report_text)
            
    def generate_html_report(self, games_analysis: Dict[str, Dict], output_file: str):
        """Generate a HTML report from analysis results.
        
        Args:
            games_analysis: Dictionary of game analyses
            output_file: File path to write HTML report to
        """
        html_start = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chess Game Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #555; margin-top: 30px; }
                .game { margin-bottom: 40px; }
                .game-info { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
                .move { margin: 10px 0; padding: 10px; border-radius: 5px; }
                .Inaccuracy { background-color: #fff3cd; }
                .Mistake { background-color: #ffe8cc; }
                .Blunder { background-color: #f8d7da; }
                .move-header { font-weight: bold; }
                .move-detail { margin-left: 20px; }
                footer { margin-top: 30px; font-size: 0.8em; color: #777; }
            </style>
        </head>
        <body>
            <h1>Chess Game Analysis Report</h1>
        """
        
        html_end = """
            <footer>
                Generated by Chess PGN Analyzer with Stockfish
            </footer>
        </body>
        </html>
        """
        
        game_html = []
        
        for game_id, game_data in games_analysis.items():
            headers = game_data['headers']
            analysis = game_data['analysis']
            
            game_html.append(f'<div class="game">')
            game_html.append(f'<h2>{game_id}</h2>')
            game_html.append(f'<div class="game-info">')
            game_html.append(f'<p>Event: {headers.get("Event", "Unknown")}</p>')
            game_html.append(f'<p>Date: {headers.get("Date", "Unknown")}</p>')
            game_html.append(f'<p>White: {headers.get("White", "Unknown")}</p>')
            game_html.append(f'<p>Black: {headers.get("Black", "Unknown")}</p>')
            game_html.append(f'<p>Result: {headers.get("Result", "Unknown")}</p>')
            game_html.append(f'</div>')
            
            if not analysis:
                game_html.append("<p>No mistakes found in this game.</p>")
                game_html.append('</div>')
                continue
            
            for move in analysis:
                if move.mistake_type:
                    # Format move notation
                    color_indicator = "." if move.player_color == chess.WHITE else "..."
                    move_notation = f"{move.move_number}{color_indicator} {move.san_move}"
                    
                    game_html.append(f'<div class="move {move.mistake_type}">')
                    game_html.append(f'<p class="move-header">Move {move_notation} ({move.mistake_type})</p>')
                    game_html.append(f'<p class="move-detail">Better move: {move.best_move_san}</p>')
                    game_html.append(f'<p class="move-detail">Eval dropped by {abs(move.eval_difference):.1f} centipawns</p>')
                    game_html.append(f'</div>')
            
            game_html.append('</div>')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_start + ''.join(game_html) + html_end)
            
        self.logger.info(f"HTML report generated at {output_file}")

    def generate_annotated_pgn(self, pgn_path: str, games_analysis: Dict[str, Dict], output_file: str):
        """Generate an annotated PGN file with mistake comments.
        
        Args:
            pgn_path: Path to the original PGN file
            games_analysis: Dictionary of game analyses
            output_file: File path to write annotated PGN to
        """
        # Read the original PGN file
        with open(pgn_path, encoding='utf-8-sig') as pgn_file:
            games = []
            game_count = 0
            
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                    
                game_count += 1
                game_id = self._get_game_identifier(game, game_count)
                
                # Get analysis for this game
                if game_id in games_analysis:
                    game_analysis = games_analysis[game_id]['analysis']
                    
                    # Convert to a dictionary for easier lookup
                    analysis_by_ply = {}
                    for move in game_analysis:
                        if move.mistake_type:
                            ply = (move.move_number - 1) * 2
                            if move.player_color == chess.BLACK:
                                ply += 1
                            analysis_by_ply[ply] = move
                    
                    # Create a new game with the same headers
                    new_game = chess.pgn.Game.from_board(chess.Board())
                    new_game.headers = game.headers
                    
                    # Add analysis comments to moves
                    ply_count = 0
                    current_node = new_game
                    node = game
                    
                    # Add the root comments if any
                    if node.comment:
                        current_node.comment = node.comment
                    
                    while node.variations:
                        node = node.variations[0]  # Follow the main line
                        move = node.move
                        
                        # Add the move to the new game
                        current_node = current_node.add_variation(move)
                        
                        # Copy the original comment if there was one
                        if node.comment:
                            current_node.comment = node.comment
                            
                        # If this move has analysis, add it to the comment
                        if ply_count in analysis_by_ply:
                            analysis = analysis_by_ply[ply_count]
                            comment = current_node.comment if current_node.comment else ""
                            
                            # Add analysis comment
                            if comment:
                                comment += " "
                            comment += f"{analysis.mistake_type}! Better: {analysis.best_move_san} ({analysis.eval_difference:.1f})"
                            
                            current_node.comment = comment
                            
                        ply_count += 1
                    
                    games.append(new_game)
                else:
                    # No analysis for this game, keep it as is
                    games.append(game)
                    
        # Write the annotated PGN
        with open(output_file, 'w', encoding='utf-8') as f:
            exporter = chess.pgn.FileExporter(f)
            for game in games:
                game.accept(exporter)
                
        self.logger.info(f"Annotated PGN file written to {output_file}")


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("chess_analyzer_main")
    
    parser = argparse.ArgumentParser(description="Chess PGN Analyzer with Stockfish")
    parser.add_argument("pgn_file", nargs="?", help="Path to the PGN file to analyze")
    parser.add_argument("--stockfish", "-s", help="Path to Stockfish executable")
    parser.add_argument("--depth", "-d", type=int, default=18, help="Analysis depth (default: 18)")
    parser.add_argument("--time", "-t", type=float, help="Time limit for analysis in seconds (overrides depth)")
    parser.add_argument("--inaccuracy", type=int, default=50, help="Threshold for inaccuracies in centipawns (default: 50)")
    parser.add_argument("--mistake", type=int, default=150, help="Threshold for mistakes in centipawns (default: 150)")
    parser.add_argument("--blunder", type=int, default=300, help="Threshold for blunders in centipawns (default: 300)")
    parser.add_argument("--output", "-o", help="Path to output text report file")
    parser.add_argument("--html", help="Path to output HTML report file")
    parser.add_argument("--annotated-pgn", help="Path to output annotated PGN file")
    parser.add_argument("--all-moves", action="store_true", help="Include all moves in reports, not just mistakes")
    parser.add_argument("--batch", help="Directory containing multiple PGN files to analyze")
    
    args = parser.parse_args()
    
    # Check if either pgn_file or batch is provided
    if not args.pgn_file and not args.batch:
        parser.error("Either a PGN file or --batch directory must be specified")
    
    try:
        # Create analyzer
        analyzer = ChessPGNAnalyzer(
            stockfish_path=args.stockfish,
            depth=args.depth,
            time_limit=args.time,
            inaccuracy_threshold=args.inaccuracy,
            mistake_threshold=args.mistake,
            blunder_threshold=args.blunder
        )
        
        # Batch mode
        if args.batch:
            process_batch(analyzer, args)
        else:
            # Single file mode
            # Analyze the PGN file
            games_analysis = analyzer.analyze_pgn_file(args.pgn_file, args.all_moves)
            
            # Generate text report
            analyzer.print_analysis_report(games_analysis, args.output)
            
            # Generate HTML report if requested
            if args.html:
                analyzer.generate_html_report(games_analysis, args.html)
                
            # Generate annotated PGN if requested
            if args.annotated_pgn:
                analyzer.generate_annotated_pgn(args.pgn_file, games_analysis, args.annotated_pgn)
                
    except ValueError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    # Success
    return 0


def process_batch(analyzer, args):
    """Process multiple PGN files in batch mode."""
    logger = logging.getLogger("chess_analyzer_batch")
    
    batch_dir = Path(args.batch)
    if not batch_dir.is_dir():
        logger.error(f"Batch directory not found: {args.batch}")
        return 1
        
    # Create output directory
    output_dir = batch_dir / "analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    # Process each PGN file
    pgn_files = list(batch_dir.glob("*.pgn"))
    logger.info(f"Found {len(pgn_files)} PGN files to analyze")
    
    for pgn_file in pgn_files:
        logger.info(f"Processing {pgn_file.name}")
        
        # Base name without extension
        base_name = pgn_file.stem
        
        try:
            # Analyze the PGN file
            games_analysis = analyzer.analyze_pgn_file(str(pgn_file), args.all_moves)
            
            # Generate text report
            text_output = output_dir / f"{base_name}_analysis.txt"
            analyzer.print_analysis_report(games_analysis, str(text_output))
            
            # Generate HTML report if requested
            if args.html:
                html_output = output_dir / f"{base_name}_analysis.html"
                analyzer.generate_html_report(games_analysis, str(html_output))
                
            # Generate annotated PGN if requested
            if args.annotated_pgn:
                pgn_output = output_dir / f"{base_name}_annotated.pgn"
                analyzer.generate_annotated_pgn(str(pgn_file), games_analysis, str(pgn_output))
                
        except Exception as e:
            logger.error(f"Error analyzing {pgn_file.name}: {e}")
            continue
    
    return 0


if __name__ == "__main__":
    exit(main())