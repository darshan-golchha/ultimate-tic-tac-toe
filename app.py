import flask
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from typing import Dict, Any

from engine import SuperTicTacToeAI

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Store game states for concurrent users
game_states: Dict[str, SuperTicTacToeAI] = {}

def start_game() -> SuperTicTacToeAI:
    """
    Initialize a new game with the AI
    """
    game = SuperTicTacToeAI()
    return game

def reset_game() -> SuperTicTacToeAI:
    """
    Reset the game state
    """
    game = SuperTicTacToeAI()
    return game

def make_move(game: SuperTicTacToeAI, move_data: Dict[str, int]) -> SuperTicTacToeAI:
    """
    Make a player move and then AI move using best move strategy
    """
    # Player move
    global_board = move_data.get('global_board')
    local_board = move_data.get('local_board')
    
    if not game.make_move(global_board, local_board, 1):
        raise ValueError("Invalid move")
    
    # Check if game is over after player move
    if game.check_global_win() != 0:
        return game
    
    # AI move using best move method
    ai_move = game.get_best_move()
    game.make_move(ai_move[0], ai_move[1], -1)
    
    return game

def get_feedback(game: SuperTicTacToeAI) -> Dict[str, Any]:
    """
    Generate game state feedback
    """
    global_win = game.check_global_win()
    feedback = {
        'global_win': global_win,
        'next_board': game.next_board,
        'possible_moves': game.get_possible_moves(),
        'sub_board_states': game.sub_board_states
    }
    return feedback


@app.route('/')
@cross_origin()
def index():
    # returning the index.html file
    return flask.send_file('index.html')

@app.route('/start', methods=['POST'])
@cross_origin()
def start():
    user_id = request.json.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
    
    # Initialize a new game for the user
    game_states[user_id] = start_game()
    return jsonify({'message': 'Game started', 'state': _serialize_game_state(game_states[user_id]),
                    'feedback': get_feedback(game_states[user_id])
                    }), 200

@app.route('/reset', methods=['POST'])
@cross_origin()
def reset():
    user_id = request.json.get('user_id')
    if not user_id or user_id not in game_states:
        return jsonify({'error': 'Invalid User ID'}), 400
    
    # Reset the user's game state
    game_states[user_id] = reset_game()
    return jsonify({'message': 'Game reset', 'state': _serialize_game_state(game_states[user_id])}), 200

@app.route('/move', methods=['POST'])
@cross_origin()
def move():
    data = request.json
    user_id = data.get('user_id')
    move_data = data.get('move')
    
    if not user_id or user_id not in game_states:
        return jsonify({'error': 'Invalid User ID'}), 400
    
    if not move_data:
        return jsonify({'error': 'Move data is required'}), 400
    
    try:
        # Process the move
        game_states[user_id] = make_move(game_states[user_id], move_data)
        feedback = get_feedback(game_states[user_id])
        return jsonify({
            'message': 'Move made', 
            'state': _serialize_game_state(game_states[user_id]), 
            'feedback': feedback
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

def _serialize_game_state(game: SuperTicTacToeAI) -> Dict[str, Any]:
    """
    Convert game state to a JSON-serializable format
    """
    return {
        'board': game.board,
        'sub_board_states': game.sub_board_states,
        'next_board': game.next_board,
        'current_player': game.current_player
    }

if __name__ == '__main__':
    app.run(debug=True)