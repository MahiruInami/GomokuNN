using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal interface IGameBoardState
    {
        public void Init(int size);
        public int GetCellState(int x, int y);
        public void SetCellState(int x, int y, int color);
        public int GetBoardSize();
        public int GetBoardStateHash();
    }
}
