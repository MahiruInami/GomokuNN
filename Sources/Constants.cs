using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static GomokuNN.Sources.Constants;

namespace GomokuNN.Sources
{
    internal class Constants
    {
        public const int DEFAULT_BOARD_SIZE = 15;
        public const int CELLS_IN_ROW_TO_WIN = 5;

        public const int CELL_SIZE = 32;
        public const int BOARD_DRAW_OFFSET = 10;

        public const int NULL_COLOR = -1;
        public const int EMPTY_COLOR = 0;
        public const int CROSS_COLOR = 1;
        public const int ZERO_COLOR = 2;
        
        public const int PLAYOUTS_TO_EXPANSION = 100;

        public enum GameResult
        {
            IN_PROGRESS,
            TIE,
            WIN,
            LOSE
        }

        public static GameResult RevertGameResult(GameResult result)
        {
            var gameResult = result;
            switch (result)
            {
                case Constants.GameResult.WIN: gameResult = Constants.GameResult.LOSE; break;
                case Constants.GameResult.LOSE: gameResult = Constants.GameResult.WIN; break;
            }

            return gameResult;
        }

        public struct MovePosition
        {
            public int X { get; set; }
            public int Y { get; set; }

            public MovePosition()
            {
                X = -1;
                Y = -1;
            }

            public MovePosition(int x, int y)
            {
                X = x;
                Y = y;
            }

            public override string ToString()
            {
                return String.Format("[{0} {1}]", X, Y);
            }
        }

        public static int RotateColor(int color)
        {
            int result = 0;
            switch (color)
            {
                case CROSS_COLOR: result = ZERO_COLOR; break;
                case ZERO_COLOR: result = CROSS_COLOR; break;
            }

            return result;
        }
    }
}
