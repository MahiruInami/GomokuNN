using Raylib_cs;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GomokuNN.Sources
{
    internal class RenderHelper
    {
        private Texture2D[]? _textures = null;
        private Texture2D[]? _helperTextures = null;

        private int _boardOffsetX = Constants.BOARD_DRAW_OFFSET;
        private int _boardOffsetY = Constants.BOARD_DRAW_OFFSET;

        public RenderHelper()
        {

        }

        public void LoadTextures()
        {
            _textures = new Texture2D[3];
            _helperTextures = new Texture2D[3];

            _textures[Constants.CROSS_COLOR] = Raylib.LoadTexture("Resources/black_piece.png");
            _textures[Constants.ZERO_COLOR] = Raylib.LoadTexture("Resources/white_piece.png");
            _textures[Constants.EMPTY_COLOR] = Raylib.LoadTexture("Resources/empty_piece.png");

            _helperTextures[Constants.CROSS_COLOR] = Raylib.LoadTexture("Resources/black_piece_ai.png");
            _helperTextures[Constants.ZERO_COLOR] = Raylib.LoadTexture("Resources/white_piece_ai.png");
            _helperTextures[Constants.EMPTY_COLOR] = Raylib.LoadTexture("Resources/empty_piece.png");
        }

        public void DrawBoardCell(int cellState, int x, int y, Color color)
        {
            if (_textures == null)
            {
                return;
            }

            Raylib.DrawTexture(_textures[cellState], x * Constants.CELL_SIZE + _boardOffsetX, y * Constants.CELL_SIZE + _boardOffsetY, color);
        }

        public void DrawAINextMoveBoardCell(int cellState, int x, int y)
        {
            if (_textures == null)
            {
                return;
            }

            Raylib.DrawTexture(_helperTextures[cellState], x * Constants.CELL_SIZE + _boardOffsetX, y * Constants.CELL_SIZE + _boardOffsetY, Color.WHITE);
        }

        public void DrawPolicyText(string value, int x, int y)
        {
            if (_textures == null)
            {
                return;
            }

            Raylib.DrawText(value, x * Constants.CELL_SIZE + _boardOffsetX + (int)(Constants.CELL_SIZE * 0.1f), y * Constants.CELL_SIZE + _boardOffsetY + (int)(Constants.CELL_SIZE * 0.0f), 8, Color.BLACK);
        }

        public void DrawWinrateText(string value, int x, int y)
        {
            if (_textures == null)
            {
                return;
            }

            Raylib.DrawText(value, x * Constants.CELL_SIZE + _boardOffsetX + (int)(Constants.CELL_SIZE * 0.1f), y * Constants.CELL_SIZE + _boardOffsetY + (int)(Constants.CELL_SIZE * 0.4f), 8, Color.BLACK);
        }

        public void DrawScoreText(string value, int x, int y)
        {
            if (_textures == null)
            {
                return;
            }

            Raylib.DrawText(value, x * Constants.CELL_SIZE + _boardOffsetX + (int)(Constants.CELL_SIZE * 0.1f), y * Constants.CELL_SIZE + _boardOffsetY + (int)(Constants.CELL_SIZE * 0.75f), 8, Color.BLACK);
        }
    }
}
