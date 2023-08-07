using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class GameHistory
    {
        public int startingColor;
        public int winningColor;
        public Constants.GameResult result;
        public List<Constants.MovePosition> moves;
    }
}
