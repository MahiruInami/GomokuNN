using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class GameBoard
    {
        IGameBoardState _state;
        int _currentTurnColor;
        Constants.GameResult _result;

        public GameBoard(IGameBoardState state, int currentTurnColor)
        {
            _state = state;
            _currentTurnColor = currentTurnColor;

            _result = Constants.GameResult.IN_PROGRESS;
        }

        public void Init(int newSize, int currentTurnColor)
        {
            _state.Init(newSize);
            _currentTurnColor = currentTurnColor;

            _result = Constants.GameResult.IN_PROGRESS;
        }

        public int GetCellState(int x, int y)
        {
            return _state.GetCellState(x, y);
        }

        public bool MakeMove(int x, int y)
        {
            if (_state.GetCellState(x, y) != Constants.EMPTY_COLOR)
            {
                return false;
            }

            _state.SetCellState(x, y, _currentTurnColor);
            _result = GameResultEstimator.EstimateResult(_state, x, y, true);
            if (_result == Constants.GameResult.IN_PROGRESS)
            {
                _currentTurnColor = Constants.RotateColor(_currentTurnColor);
            }

            return true;
        }

        public int GetBoardSize()
        {
            return _state.GetBoardSize();
        }

        public int GetCurrentTurnColor()
        {
            return _currentTurnColor;
        }

        public void SetCurrentTurnColor(int value)
        {
            _currentTurnColor = value;
        }

        public IGameBoardState GetBoardState()
        {
            return _state;
        }

        public Constants.GameResult GetGameState()
        {
            return _result;
        }
    }
}
