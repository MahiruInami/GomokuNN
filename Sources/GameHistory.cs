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

        public static GameHistory CreateFromPSQFile(string path)
        {
            var gameHistory = new GameHistory();
            gameHistory.moves = new List<Constants.MovePosition>();

            int currentColor = 1;
            var gameFile = File.ReadAllLines(path);
            for (int i = 0; i < gameFile.Length; i++)
            {
                var move = gameFile[i].Split(",");
                if (move == null)
                {
                    continue;
                }

                if (i == 0)
                {
                    int startingColor = int.Parse(move[move.Length - 1]);
                    gameHistory.startingColor = startingColor + 1;
                }

                if (i > 0 && i <= gameFile.Length - 5)
                {
                    int x = int.Parse(move[0]) - 1;
                    int y = int.Parse(move[1]) - 1;

                    gameHistory.moves.Add(new Constants.MovePosition(x, y));
                    if (i < gameFile.Length - 5)
                    {
                        currentColor = Constants.RotateColor(currentColor);
                    }
                }

                if (i == gameFile.Length - 1)
                {
                    int winningColor = int.Parse(move[0]);
                    if (winningColor != 0)
                    {
                        gameHistory.winningColor = currentColor;
                        gameHistory.result = currentColor == gameHistory.startingColor ? Constants.GameResult.WIN : Constants.GameResult.LOSE;
                    }
                    else
                    {
                        gameHistory.winningColor = 0;
                        gameHistory.result = Constants.GameResult.TIE;
                    }
                }
            }

            return gameHistory;
        }
    }
}
