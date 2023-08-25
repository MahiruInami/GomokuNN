using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources.Estimators
{
    internal interface IGameEstimator
    {
        public void InitFromState(IGameBoardState state, int initialColor, int estimatorColor);
        public bool SelectNextNode(int x, int y, int color, bool cleanUpTree = false);

        public void StartEstimation();
        public bool IsEstimationInProgress();
        public void StopEstimation();
        public bool HasContiniousEstimationSupport();
        public void EstimateOnce();

        public int GetCurrentPlayoutsCount();
        public Constants.MovePosition GetBestMove();

        public void SetExplorationConst(float value);
        public float GetMoveProbability(int x, int y);

        public List<TrainingSample> GetTrainingSamples(int winnerColor, ref HashSet<long> knownPositions);
        public void FillTrainingSamples(ref List<TrainingSample> samples, int winnerColor, ref HashSet<long> knownPositions);

        public void OnDrawCell(int x, int y, ref RenderHelper renderHelper);
        public void DebugMenuDraw(ref GameBoard gameBoard);
    }
}
