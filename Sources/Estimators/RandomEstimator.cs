using ImGuiNET;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources.Estimators
{
    internal class RandomEstimator : IGameEstimator
    {

        ArrayGameBoardState _state = new ArrayGameBoardState(3);
        IncrementalMovesPolicy _policy;
        float _mctsExplorationConst = 1.0f;

        Random _rndGenerator = new Random();

        public void InitFromState(IGameBoardState gameState, int turnColor, int estimatorColor)
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

        public bool HasContiniousEstimationSupport()
        {
            return false;
        }

        public void EstimateOnce()
        {
        }

        public void StartEstimation()
        {
        }

        public bool IsEstimationInProgress()
        {
            return false;
        }

        public void StopEstimation()
        {
        }

        public void SetExplorationConst(float value)
        {
            _mctsExplorationConst = value;
        }

        public void OnDrawCell(int x, int y, ref RenderHelper renderHelper)
        {

        }

        public void DebugMenuDraw(ref GameBoard gameBoard)
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

        public List<TrainingSample> GetTrainingSamples(int winnerColor, ref HashSet<long> knownPositions)
        {
            var samples = new List<TrainingSample>();
            FillTrainingSamples(ref samples, winnerColor, ref knownPositions);

            return samples;
        }

        public void FillTrainingSamples(ref List<TrainingSample> samples, int winnerColor, ref HashSet<long> knownPositions)
        {

        }
    }
}
