using ImGuiNET;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class RandomEstimator : IGameEstimator
    {

        ArrayGameBoardState _state = new ArrayGameBoardState(3);
        IncrementalMovesPolicy _policy;
        float _mctsExplorationConst = 1.0f;

        Random _rndGenerator = new Random();

        public void InitFromState(IGameBoardState gameState, int turnColor)
        {
            var boardSize = gameState.GetBoardSize();
            _state.Init(boardSize);
            for (int x = 0; x < boardSize; x++)
            {
                for (int y = 0; y < boardSize; y++)
                {
                    int state = gameState.GetCellState(x, y);
                    _state.SetCellState(x, y, state);
                }
            }

            _policy = new IncrementalMovesPolicy(5);
            _policy.Init(_state);
        }

        public bool SelectNextNode(int x, int y, int color, bool cleanUpTree)
        {
            _state.SetCellState(x, y, color);
            IGameBoardState state = _state;
            _policy.Update(x, y, ref state);

            return true;
        }

        public void Estimate(int simulationsCount)
        {
        }

        public void SetExplorationConst(float value)
        {
            _mctsExplorationConst = value;
        }

        public void DebugMenuDraw(ref GameBoard gameBoard)
        {

        }

        public void DebugFieldDraw()
        {

        }

        public int GetCurrentPlayoutsCount()
        {
            return int.MaxValue;
        }

        public Constants.MovePosition GetBestMove()
        {
            var availableMoves = _policy.GetHashedPositions();
            if (availableMoves.Count == 0)
            {
                return new Constants.MovePosition();
            }
            int rndIndex = _rndGenerator.Next(availableMoves.Count);
            return new Constants.MovePosition(_policy.GetUnhashedPositionX(availableMoves[rndIndex]), _policy.GetUnhashedPositionY(availableMoves[rndIndex]));
        }

        public float GetMoveProbability(int x, int y)
        {
            return 0.0f;
        }

        public List<TrainingSample> GetTrainingSamples(int winnerColor)
        {
            var samples = new List<TrainingSample>();
            FillTrainingSamples(ref samples, winnerColor);

            return samples;
        }

        public void FillTrainingSamples(ref List<TrainingSample> samples, int winnerColor)
        {

        }
    }
}
