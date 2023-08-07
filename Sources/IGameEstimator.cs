using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal interface IGameEstimator
    {
        public void InitFromState(IGameBoardState state, int initialColor);
        public bool SelectNextNode(int x, int y, int color, bool cleanUpTree = false);
        public void Estimate(int simulationsCount);

        public int GetCurrentPlayoutsCount();
        public Constants.MovePosition GetBestMove();

        public void SetExplorationConst(float value);
        public float GetMoveProbability(int x, int y);

        public void DebugMenuDraw(ref GameBoard gameBoard);
        public void DebugFieldDraw();
    }
}
